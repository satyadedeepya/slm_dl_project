"""Deep Learning generator checks.

Every reference solution must pass the grader, test parameters must differ from training ones, and
(when PyTorch is installed) answers are recomputed independently with torch layers and autograd.
"""
import random
import re
import unittest

from data.generators import dl_gen
from data.generators.grading import grade

try:
    import torch
except ImportError:  # the generators themselves need only the standard library
    torch = None

NUM = r"-?\d+(?:\.\d+)?"


def nums(text: str) -> list[float]:
    return [float(x) for x in re.findall(NUM, text)]


def sample(gen, split="train", n=40, seed=0):
    rng = random.Random(seed)
    return [gen(rng, split) for _ in range(n)]


class TestDLGenerators(unittest.TestCase):
    def test_reference_solutions_pass_grader(self):
        for split in ("train", "test"):
            for rec in dl_gen.generate(300, seed=1, split=split):
                self.assertTrue(grade(rec["final"], rec["messages"][2]["content"]), rec["messages"][1]["content"])

    def test_train_and_test_questions_disjoint(self):
        train = {r["messages"][1]["content"] for r in dl_gen.generate(600, seed=42)}
        test = {r["messages"][1]["content"] for r in dl_gen.generate(100, seed=2026, split="test")}
        self.assertFalse(train & test)

    def test_alexnet_conv1_by_hand(self):
        # 227x227x3, 96 filters 11x11, stride 4, no padding -> 55x55x96, (11*11*3+1)*96 = 34944
        self.assertEqual((227 - 11) // 4 + 1, 55)
        self.assertEqual((11 * 11 * 3 + 1) * 96, 34944)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestDLAgainstTorch(unittest.TestCase):
    def test_conv(self):
        for split in ("train", "test"):
            for p in sample(dl_gen.gen_conv, split, n=15):
                h, _, c, f, k, _, s, pad = nums(p.question)[:8]
                conv = torch.nn.Conv2d(int(c), int(f), int(k), stride=int(s), padding=int(pad))
                out = conv(torch.zeros(1, int(c), int(h), int(h)))
                n_params = sum(t.numel() for t in conv.parameters())
                self.assertEqual(p.final["values"], [out.shape[2], out.shape[3], out.shape[1], n_params])

    def test_mlp_params(self):
        for p in sample(dl_gen.gen_mlp_params, n=30):
            sizes = [int(x) for x in re.search(r"sizes ([\d → ]+)", p.question).group(1).split(" → ")]
            layers = [torch.nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])]
            self.assertEqual(p.final["values"][0], sum(t.numel() for l in layers for t in l.parameters()))

    def test_softmax_cross_entropy(self):
        for p in sample(dl_gen.gen_softmax_ce, "test", n=30):
            z = torch.tensor(nums(p.question.split("logits z = ")[1].split(")")[0]))
            t = int(re.search(r"true class is class (\d)", p.question).group(1)) - 1
            probs = torch.softmax(z, 0).tolist()
            loss = torch.nn.functional.cross_entropy(z[None], torch.tensor([t])).item()
            for got, want in zip(p.final["values"], probs + [loss]):
                self.assertAlmostEqual(got, want, places=3)

    def test_gd_and_backprop_with_autograd(self):
        for p in sample(dl_gen.gen_gd_linear, "test", n=20):
            w0, b0, x, y, lr = nums(p.question.replace("L = (ŷ − y)²", ""))[:5]
            w = torch.tensor(w0, requires_grad=True)
            b = torch.tensor(b0, requires_grad=True)
            ((w * x + b - y) ** 2).backward()
            want = [w.grad.item(), b.grad.item(), w0 - lr * w.grad.item(), b0 - lr * b.grad.item()]
            for got, exp in zip(p.final["values"], want):
                self.assertAlmostEqual(got, exp, places=3)
        for p in sample(dl_gen.gen_backprop_sigmoid, "test", n=20):
            q = re.sub(r"[₁₂]", "", p.question)
            w1, w2, b = nums(q.split("where ")[1].split(". For input")[0])
            x1, x2 = nums(q.split("input (x, x) = (")[1].split(")")[0])
            t = float(re.search(r"target t = (\d)", q).group(1))
            ws = torch.tensor([w1, w2], requires_grad=True)
            a = torch.sigmoid(ws[0] * x1 + ws[1] * x2 + b)
            (0.5 * (a - t) ** 2).backward()
            for got, exp in zip(p.final["values"], ws.grad.tolist()):
                self.assertAlmostEqual(got, exp, places=4)

    def test_attention(self):
        for p in sample(dl_gen.gen_attention, "test", n=20):
            vecs = [nums(v) for v in re.findall(r"\(([^)]*)\)", p.question)[:4]]
            q, k = torch.tensor(vecs[0]), torch.tensor(vecs[1:4])
            want = torch.softmax(k @ q / 2 ** 0.5, 0).tolist()
            for got, exp in zip(p.final["values"], want):
                self.assertAlmostEqual(got, exp, places=3)


if __name__ == "__main__":
    unittest.main()
