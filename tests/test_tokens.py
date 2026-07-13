from project_main.tokens import build_vocab

def test_vocab_size_matches_config():
    vocab = build_vocab({"vocab_size": 130})
    assert vocab.size == 130


def test_special_tokens():
    vocab = build_vocab({"vocab_size": 130})
    assert vocab.bos_id == 0
    assert vocab.newline_id == 1

def test_encode_decode():
    vocab = build_vocab({"vocab_size": 30})
    for i in range(vocab.size):
        token = vocab.decode_token(i)
        token_id = vocab.encode_token(token)
        assert token_id == i
