"""Which parameters of the conditional generator actually receive a gradient?

    python scripts/check_generator_gradients.py

Builds the generator exactly as train_pairs.py does, runs one forward and backward on
synthetic molecules, and lists every parameter whose gradient is None -- that is, every
parameter that is allocated, counted in the model's size, and never trained because it
sits outside the computation graph.

The reason to check. The decoder conditions on HybridMoLA.encode_for_generation, which
returns the encoder's UNPOOLED per-atom and per-character states and discards fused_all,
the stacked one-token-per-layer-per-modality sequence that MoLA's cross-layer attention
operates on. Cross-attention needs a sequence to attend over and a single pooled vector
would give the decoder one position, so using the raw states is the right call for
generation. But it means the fusion module itself may never run on this path, and if so
the modality ablation is comparing which raw states enter a concatenated memory rather
than anything about MoLA's fusion -- which would explain a null result without the fusion
having been tested at all.

That is a claim about the computation graph, and this settles it by measurement rather
than by reading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torch_geometric.data import Batch, Data  # noqa: E402

from crossmodal_model.generation.conditional_decoder import ConditionalMoleculeGenerator  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402

HIDDEN, LAYERS, VOCAB, SM_VOCAB, MAX_SM, N_BINS, N_PROPS = 256, 3, 40, 40, 100, 21, 4


def fake_molecule(gen: torch.Generator, n_atoms: int) -> Data:
    e = torch.stack([torch.arange(n_atoms - 1), torch.arange(1, n_atoms)])
    return Data(
        x=torch.randint(0, 5, (n_atoms, 9), generator=gen),
        edge_index=torch.cat([e, e.flip(0)], dim=1),
        edge_attr=torch.randint(0, 3, (2 * (n_atoms - 1), 3), generator=gen),
        sm=torch.randint(1, SM_VOCAB, (1, MAX_SM), generator=gen),
    )


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    mola = HybridMoLA(
        sm_vocab_size=SM_VOCAB, hidden_dim=HIDDEN, output_dim=1, num_layers=LAYERS,
        positional_smiles=True, max_sm_len=MAX_SM, use_graph=True, use_smiles=True,
        gin_hidden_mult=8,
    )
    model = ConditionalMoleculeGenerator(
        mola, vocab_size=VOCAB, hidden_dim=HIDDEN, pad_idx=0,
        cond_vocab_sizes=[N_BINS] * N_PROPS, cond_null_bins=[N_BINS - 1] * N_PROPS,
        cond_dropout=0.15, decoder_layers=6, max_len=MAX_SM + 32,
    )
    model.train()

    batch = Batch.from_data_list([fake_molecule(gen, 8 + i) for i in range(4)])
    cond = torch.randint(0, N_BINS, (4, N_PROPS))
    dec_in = torch.randint(1, VOCAB, (4, 12))
    dec_tgt = torch.randint(1, VOCAB, (4, 12))

    logits = model(batch, dec_in, cond)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))
    loss.backward()

    trained, dead = [], []
    for name, p in model.named_parameters():
        (trained if p.grad is not None else dead).append((name, p.numel()))

    n_t, n_d = sum(n for _, n in trained), sum(n for _, n in dead)
    print(f"  parametros con gradiente : {n_t:>12,}  ({len(trained)} tensores)")
    print(f"  parametros SIN gradiente : {n_d:>12,}  ({len(dead)} tensores)")
    print(f"  total                    : {n_t + n_d:>12,}")

    if dead:
        print(f"\n  {n_d / (n_t + n_d):.1%} del modelo esta asignado y nunca se entrena:")
        for name, n in sorted(dead, key=lambda kv: -kv[1]):
            print(f"    {name:<52} {n:>10,}")
        roots = sorted({n.split(".")[1] for n, _ in dead if n.startswith("mola.")})
        print(f"\n  modulos de mola afectados: {roots}")
        print("\n  Si ahi aparecen cross_attention, layer_weights u out_layer_final, la")
        print("  fusion entre capas de MoLA no interviene en la generacion: el decoder")
        print("  condiciona sobre los estados crudos por atomo y por caracter, y la")
        print("  ablacion de modalidades compara que estados entran en la memoria, no la")
        print("  fusion. Un resultado nulo ahi no dice nada sobre la fusion.")
    else:
        print("\n  Todos los parametros reciben gradiente: la fusion si interviene.")


if __name__ == "__main__":
    main()
