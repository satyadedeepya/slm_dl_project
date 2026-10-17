"""Checks the CN and DBMS solvers against textbook examples, and the grader on
every generated reference solution.

Run: python -m unittest discover -s tests -t . -v
"""
import unittest

from data.generators import cn_gen, dbms_gen
from data.generators.cn_gen import crc_divide, ones_complement_sum, subnet_info
from data.generators.dbms_gen import candidate_keys, closure, normal_form
from data.generators.grading import grade


def F(spec: str):
    """'AB->C, C->D' -> list of (frozenset, frozenset)."""
    out = []
    for part in spec.split(","):
        lhs, rhs = part.split("->")
        out.append((frozenset(lhs.strip()), frozenset(rhs.strip())))
    return out


class TestSubnetting(unittest.TestCase):
    def test_slash26(self):
        i = subnet_info("192.168.1.130", 26)
        self.assertEqual((i["network"], i["broadcast"], i["hosts"], i["mask"]),
                         ("192.168.1.128", "192.168.1.191", 62, "255.255.255.192"))

    def test_slash21(self):
        i = subnet_info("172.16.45.200", 21)
        self.assertEqual((i["network"], i["broadcast"], i["hosts"]), ("172.16.40.0", "172.16.47.255", 2046))


class TestCRC(unittest.TestCase):
    def test_forouzan(self):  # dataword 1001, divisor 1011 -> remainder 110
        self.assertEqual(crc_divide("1001000", "1011")[0], "110")

    def test_tanenbaum(self):  # frame 1101011111, G = 10011 -> transmitted 11010111110010
        rem = crc_divide("1101011111" + "0000", "10011")[0]
        self.assertEqual("1101011111" + rem, "11010111110010")

    def test_receiver_zero_remainder_on_valid_frame(self):
        self.assertEqual(crc_divide("11010111110010", "10011")[0], "0000")


class TestChecksum(unittest.TestCase):
    def test_ipv4_header_example(self):  # Wikipedia IPv4 header checksum example -> B861
        words = [0x4500, 0x0073, 0x0000, 0x4000, 0x4011, 0x0000, 0xC0A8, 0x0001, 0xC0A8, 0x00C7]
        total, _ = ones_complement_sum(words)
        self.assertEqual((~total) & 0xFFFF, 0xB861)


class TestFDAlgorithms(unittest.TestCase):
    R4 = frozenset("ABCD")

    def test_closure(self):
        fds = F("A->B, B->C, CD->E")
        self.assertEqual(closure(frozenset("A"), fds)[0], frozenset("ABC"))
        self.assertEqual(closure(frozenset("AD"), fds)[0], frozenset("ABCDE"))

    def test_candidate_keys_cyclic(self):  # classic: AB->C, C->D, D->A gives keys AB, BC, BD
        keys = candidate_keys(self.R4, F("AB->C, C->D, D->A"))
        self.assertEqual(set(keys), {frozenset("AB"), frozenset("BC"), frozenset("BD")})

    def test_normal_forms(self):
        self.assertEqual(normal_form(self.R4, F("AB->C, C->D, D->A"))[0], "3NF")   # all attributes prime
        self.assertEqual(normal_form(self.R4, F("AB->C, B->D"))[0], "1NF")          # partial dependency
        self.assertEqual(normal_form(self.R4, F("A->B, B->C, C->D"))[0], "2NF")     # transitive
        self.assertEqual(normal_form(self.R4, F("A->BCD"))[0], "BCNF")
        self.assertEqual(normal_form(frozenset("ABC"), F("AB->C, C->B"))[0], "3NF")  # textbook 3NF-not-BCNF

    def test_lossless_rule(self):
        fds = F("A->B")
        c = closure(frozenset("A"), fds)[0]
        self.assertTrue(frozenset("AB") <= c)       # R1(A,B), R2(A,C): lossless
        c = closure(frozenset("B"), fds)[0]
        self.assertFalse(frozenset("AB") <= c or frozenset("BC") <= c)  # R1(A,B), R2(B,C): lossy


class TestGrader(unittest.TestCase):
    def test_tokens(self):
        final = {"type": "tokens", "values": ["192.168.1.128", "192.168.1.191", "62"]}
        self.assertTrue(grade(final, "Final answer: Network 192.168.1.128, broadcast 192.168.1.191, 62 hosts."))
        self.assertFalse(grade(final, "Final answer: Network 192.168.1.128, broadcast 192.168.1.191, 64 hosts"))
        crc = {"type": "tokens", "values": ["110", "1001110"]}
        self.assertTrue(grade(crc, "Final answer: remainder 110, frame 1001110"))
        self.assertFalse(grade(crc, "Final answer: frame 1001110"))  # '110' inside the frame does not count
        hexa = {"type": "tokens", "values": ["B861"]}
        self.assertTrue(grade(hexa, "Final answer: checksum = 0xb861"))

    def test_relative_numbers(self):
        final = {"type": "numbers", "values": [1.2, 10, 11.2], "rel_tol": 0.01}
        self.assertTrue(grade(final, "Final answer: 1.2 ms, 10 ms, 11.2 ms"))
        self.assertFalse(grade(final, "Final answer: 0.0012 s, 10 ms, 11.2 ms"))

    def test_sets_keys_choice(self):
        attrs = list("ABCDE")
        self.assertTrue(grade({"type": "attrset", "values": list("ABC"), "attrs": attrs},
                              "Final answer: A+ = {A, B, C}"))
        self.assertTrue(grade({"type": "attrset", "values": list("ABC"), "attrs": attrs}, "Final answer: A+ = ABC"))
        self.assertFalse(grade({"type": "attrset", "values": list("ABC"), "attrs": attrs}, "Final answer: A+ = {A, B}"))
        keys = {"type": "keysets", "values": [["A", "B"], ["B", "C"]], "attrs": attrs}
        self.assertTrue(grade(keys, "Final answer: Candidate keys = {A, B}, {B, C}"))
        self.assertTrue(grade(keys, "Final answer: Candidate keys: AB, BC"))
        self.assertFalse(grade(keys, "Final answer: Candidate keys: AB"))
        nf = {"type": "choice", "value": "3NF", "options": ["1NF", "2NF", "3NF", "BCNF"]}
        self.assertTrue(grade(nf, "Final answer: 3NF"))
        self.assertTrue(grade(nf, "Final answer: it is not in BCNF, so the highest is 3NF"))
        self.assertFalse(grade(nf, "Final answer: BCNF"))

    def test_every_reference_solution_passes(self):
        for mod in (cn_gen, dbms_gen):
            for split in ("train", "test"):
                for rec in mod.generate(1500, seed=11, split=split):
                    self.assertTrue(grade(rec["final"], rec["messages"][-1]["content"]), rec["id"])

    def test_answer_balance(self):
        recs = dbms_gen.generate(1500, seed=3)
        nfs = [r["final"]["value"] for r in recs if r["topic"] == "normalization/highest_nf"]
        for nf in dbms_gen.NF_ORDER:
            self.assertGreater(nfs.count(nf) / len(nfs), 0.15, nf)
        ll = [r["final"]["value"] for r in recs if r["topic"] == "decomposition/lossless"]
        self.assertGreater(ll.count("lossless") / len(ll), 0.35)
        self.assertGreater(ll.count("lossy") / len(ll), 0.35)


if __name__ == "__main__":
    unittest.main()
