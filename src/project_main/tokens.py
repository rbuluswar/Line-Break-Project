from dataclasses import dataclass


SPECIAL_TOKENS = ("BOS", "NEWLINE", "LONG_A", "LONG_B")


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def bos_id(self) -> int:
        return self.token_to_id["BOS"]
    
    @property
    def newline_id(self) -> int:
        return self.token_to_id["NEWLINE"]

    def encode_token(self, token: str) -> int:
        return self.token_to_id[token]

    def decode_token(self, token_id: int) -> str:
        return self.id_to_token[int(token_id)]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.encode_token(token) for token in tokens]

    def decode(self, token_ids: list[int]) -> list[str]:
        return [self.decode_token(token_id) for token_id in token_ids]



def build_vocab(task_cfg: dict) -> Vocab:
    print("Building vocab...")
    print(SPECIAL_TOKENS)

    vocab_size = task_cfg["vocab_size"]
    num_token_lengths = task_cfg["num_token_lengths"]

    num_special = len(SPECIAL_TOKENS)
    num_regular = vocab_size - num_special

    if num_regular <= 0:
        raise ValueError(
            f"vocab_size={vocab_size} is too small. "
            f"It must be greater than {num_special}."
        )
    
    if num_regular % num_token_lengths != 0:
        raise ValueError(
            f"num_regular={num_regular} is not divisible by {num_token_lengths}. "
            f"It must be divisible by {num_token_lengths}."
        )

    tokens: list[str] = []
    tokens += SPECIAL_TOKENS

    #Create Tokens of Length 1 to num_token_lengths
    for j in range(num_token_lengths):
        tokens += [f"TOKEN_{i}_{j+1}" for i in range(num_regular//num_token_lengths)]


    if len(tokens) != vocab_size:
        raise ValueError(f"Expected vocab_size={vocab_size}, got {len(tokens)} tokens.")

    if len(tokens) != len(set(tokens)):
        raise ValueError("Vocabulary contains duplicate tokens.")

    token_to_id = {token: i for i, token in enumerate(tokens)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )

