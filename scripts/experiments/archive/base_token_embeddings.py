from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    description="Do PCA over tok embs and plot selected toks"
)

parser.add_argument(
    "--no-ints", action="store_true", help="Doesn't plot ints if passed"
)

args = parser.parse_args()

REMOVE_NUMBERS: bool = True

gen_path = Path(
    "/Users/joey/research/miles/reasoning-trajectory-private/runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/generation/generations.jsonl"
)
used_token_ids = set()

with gen_path.open() as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        used_token_ids.update(row["generated_token_ids"])

model_name = "mlx-community/SmolLM3-3B-bf16"

tok = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)

E = model.get_input_embeddings().weight.detach().float().cpu().numpy()

pca = PCA(n_components=4)
X = pca.fit_transform(E)


def is_int_token(s: str) -> bool:
    try:
        int(s.strip())
        return True
    except ValueError:
        return False


tokens = [tok.decode([i]) for i in range(E.shape[0])]

df = pd.DataFrame(
    {
        "token_id": range(E.shape[0]),
        "token": tokens,
        "pc1": X[:, 0],
        "pc2": X[:, 1],
        "pc3": X[:, 2],
        "pc4": X[:, 3],
    }
)

if args.no_ints:
    df = df[~df["token"].apply(is_int_token)].copy()
df_used = df[df["token_id"].isin(used_token_ids)].copy()


print(f"Used {len(df_used)} unique tokens out of vocab size {len(df)}")

df_small = df_used.sample(min(10000, len(df_used)), random_state=0)

fig = px.scatter_3d(
    df_small,
    x="pc1",
    y="pc2",
    z="pc3",
    color="pc4",
    color_continuous_scale="Turbo",  # high contrast
    hover_data=["token_id", "token"],
)
# control point size
fig.update_traces(marker=dict(size=2))

fig.show()
