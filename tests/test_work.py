from vramcalc import Arch

from servcalc import work


def test_param_splits(tiny):
    assert work.p_mm(tiny) == 90_432          # P - V*h (untied input embedding is a lookup)
    assert work.p_body(tiny) == 74_048        # P_mm - V*h (lm_head)
    assert work.p_lin(tiny) == 73_728


def test_weight_bytes_bf16(tiny):
    w = work.weight_bytes(tiny, 16)
    assert (w.resident, w.stream) == (213_632, 180_864)


def test_weight_bytes_quantized_keeps_other_tensors_16bit(tiny):
    w = work.weight_bytes(tiny, 4)
    assert w.resident == 73_728 * 4 // 8 + (106_816 - 73_728) * 2
    assert w.stream == w.resident - 256 * 64 * 2


def test_weight_bytes_override(tiny):
    w = work.weight_bytes(tiny, 4, override=100_000)
    assert w.resident == 100_000
    assert w.stream == 100_000 - 32_768
    assert work.weight_bytes(tiny, 4, override=10).stream == 0   # clamped


def test_weight_bytes_tied(tiny_cfg):
    tied = Arch.from_config({**tiny_cfg, "tie_word_embeddings": True})
    w = work.weight_bytes(tied, 16)
    assert w.stream == w.resident


def test_kv_bytes_per_token(tiny):
    assert work.kv_bytes_per_token(tiny, 16) == 256
    assert work.kv_bytes_per_token(tiny, 8) == 128


def test_decode_step(tiny):
    # N=2, mean ctx 10.5: bytes = 180864 + 256*(2*10.5 + 2) = 186752
    # flops = 2*90432*2 + 4*2*64*2*10.5 = 372480
    w = work.decode_step(tiny, n=2, sum_ctx=21.0, kv_tok=256, w_stream=180_864, kv_realloc=False)
    assert (w.bytes, w.flops) == (186_752.0, 372_480.0)


def test_decode_step_weight_part_is_n_independent(tiny):
    a = work.decode_step(tiny, 1, 10.0, 256, 180_864, False)
    b = work.decode_step(tiny, 4, 40.0, 256, 180_864, False)
    assert b.bytes - a.bytes == 256 * (40 - 10) + 256 * 3   # only KV read + write grow


def test_decode_step_kv_realloc(tiny):
    a = work.decode_step(tiny, 2, 21.0, 256, 180_864, False)
    b = work.decode_step(tiny, 2, 21.0, 256, 180_864, True)
    assert b.bytes - a.bytes == 2 * 256 * 21.0
    assert b.flops == a.flops


def test_prefill_chunk(tiny):
    # C=4, c0=4, not last: bytes = 256*(4+4) = 2048; flops = 2*74048*4 + 4*2*64*(16 + 10) = 605696
    w = work.prefill_chunk(tiny, c_len=4, prior_ctx=4, last=False, kv_tok=256)
    assert (w.bytes, w.flops) == (2048.0, 605_696.0)
    last = work.prefill_chunk(tiny, 4, 4, True, 256)
    assert last.flops == 605_696.0 + 2 * 256 * 64


def test_prefill_chunks_and_conservation(tiny):
    assert work.prefill_chunks(8, 4) == [(4, 0, False), (4, 4, True)]
    assert work.prefill_chunks(8, 6) == [(6, 0, False), (2, 6, True)]
    assert work.prefill_chunks(8, None) == [(8, 0, True)]
    total = work.prefill_total_flops(tiny, 8)   # 2*74048*8 + 2*256*64 + 2*2*64*8*9
    assert total == 1_235_968.0
    for chunk in (1, 3, 4, 8, None):
        s = sum(work.prefill_chunk(tiny, c, c0, last, 256).flops
                for c, c0, last in work.prefill_chunks(8, chunk))
        assert s == total


def test_work_add():
    assert work.Work(1.0, 2.0) + work.Work(3.0, 4.0) == work.Work(4.0, 6.0)
