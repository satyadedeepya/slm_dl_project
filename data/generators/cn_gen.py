"""Verified Computer Networks problem generator.

Problem families (answers computed, never written by an LLM):
  * Subnetting: analyse an address/prefix; divide a network into subnets
  * CRC: sender remainder + transmitted frame; receiver check
  * Internet checksum (16-bit one's complement)
  * Delays: transmission + propagation, store-and-forward over several links
  * Link utilisation: stop-and-wait, Go-Back-N / Selective Repeat windows

Usage:
  python -m data.generators.cn_gen --n 900 --seed 42 --out data/processed/gen_cn_train.jsonl
"""
from __future__ import annotations

import ipaddress
import math
import random

from .common import Problem, cli, run_generators

# =====================================================================
# Subnetting
# =====================================================================


def subnet_info(ip: str, prefix: int) -> dict:
    net = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
    hosts = max(net.num_addresses - 2, 0)
    return {
        "network": str(net.network_address),
        "broadcast": str(net.broadcast_address),
        "mask": str(net.netmask),
        "first": str(net.network_address + 1),
        "last": str(net.broadcast_address - 1),
        "hosts": hosts,
    }


def _interesting_octet(prefix: int) -> int:
    """Index (0-3) of the octet where the network/host boundary falls."""
    return min(prefix // 8, 3)


def gen_subnet_analyse(rng: random.Random, split: str = "train") -> Problem:
    prefix = rng.randint(17, 28) if split == "train" else rng.choice([9, 12, 14, 19, 22, 25, 27, 29, 30])
    first = rng.choice([10, 172, 192, rng.randint(1, 223)])
    ip = f"{first}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    info = subnet_info(ip, prefix)

    k = _interesting_octet(prefix)
    mask_oct = int(info["mask"].split(".")[k])
    block = 256 - mask_oct
    ip_oct = int(ip.split(".")[k])
    q = rng.choice([
        f"A host has IP address {ip}/{prefix}. Find the subnet mask, network address, broadcast address, "
        "first and last usable host addresses, and the number of usable hosts.",
        f"For the address {ip} with prefix length /{prefix}, determine the network address, broadcast address, "
        "usable host range and number of usable hosts.",
        f"Given {ip}/{prefix}, which subnet does the host belong to? Give the network and broadcast addresses, "
        "the range of valid host addresses and how many hosts the subnet supports.",
    ])
    octet_name = ["first", "second", "third", "fourth"][k]
    sol = (
        f"**Subnet mask:** /{prefix} → {info['mask']}\n\n"
        f"The network/host boundary falls in the {octet_name} octet (mask octet = {mask_oct}), so the block size is "
        f"256 − {mask_oct} = {block}.\n\n"
        f"The {octet_name} octet of the address is {ip_oct}; ⌊{ip_oct} / {block}⌋ × {block} = {ip_oct // block * block}, "
        f"so the subnet starts there (all host bits = 0) and ends {block - 1} later in that octet "
        "(all host bits = 1).\n\n"
        f"- Network address: **{info['network']}**\n"
        f"- Broadcast address: **{info['broadcast']}**\n"
        f"- First usable host: {info['first']}\n"
        f"- Last usable host: {info['last']}\n"
        f"- Usable hosts = 2^(32 − {prefix}) − 2 = 2^{32 - prefix} − 2 = {info['hosts']}\n\n"
        f"**Final answer:** Mask = {info['mask']}, Network = {info['network']}, Broadcast = {info['broadcast']}, "
        f"Hosts = {info['first']} to {info['last']}, Usable hosts = {info['hosts']}"
    )
    return Problem("CN", "subnetting/analyse", q, sol,
                   {"type": "tokens", "values": [info["network"], info["broadcast"], str(info["hosts"])]})


def gen_subnet_divide(rng: random.Random, split: str = "train") -> Problem:
    base_prefix = rng.choice([16, 20, 22, 24]) if split == "train" else rng.choice([18, 21, 23, 24, 25])
    n_subnets = rng.randint(3, 12) if split == "train" else rng.randint(5, 30)
    bits = math.ceil(math.log2(n_subnets))
    new_prefix = base_prefix + bits
    if new_prefix > 30:
        new_prefix, bits = 30, 30 - base_prefix
        n_subnets = 2 ** bits
    base = ipaddress.ip_network(
        f"{rng.choice([10, 172, 192])}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.0/{base_prefix}", strict=False)
    subnets = list(base.subnets(new_prefix=new_prefix))
    k = rng.randint(2, min(len(subnets), n_subnets))
    target = subnets[k - 1]
    hosts = target.num_addresses - 2
    ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")
    q = (f"The network {base} must be divided into at least {n_subnets} equal-sized subnets. "
         f"What is the new subnet mask, how many usable hosts does each subnet have, and what are the network "
         f"and broadcast addresses of the {ordinal} subnet (counting from 1)?")
    sol = (
        f"Subnet bits needed: 2^b ≥ {n_subnets} → b = {bits} (2^{bits} = {2 ** bits} subnets).\n\n"
        f"New prefix = /{base_prefix} + {bits} = /{new_prefix} → mask **{target.netmask}**.\n\n"
        f"Each subnet has 2^{32 - new_prefix} = {target.num_addresses} addresses, so usable hosts = "
        f"{target.num_addresses} − 2 = **{hosts}**.\n\n"
        f"Subnets in order: " + ", ".join(str(s) for s in subnets[:min(k + 1, len(subnets))])
        + (" ..." if k + 1 < len(subnets) else "") + "\n\n"
        f"The {ordinal} subnet is {target}: network **{target.network_address}**, broadcast "
        f"**{target.broadcast_address}**.\n\n"
        f"**Final answer:** Mask = {target.netmask} (/{new_prefix}), {hosts} usable hosts per subnet, "
        f"{ordinal} subnet: Network = {target.network_address}, Broadcast = {target.broadcast_address}"
    )
    return Problem("CN", "subnetting/divide", q, sol,
                   {"type": "tokens", "values": [str(target.netmask), str(hosts),
                                                 str(target.network_address), str(target.broadcast_address)]})


# =====================================================================
# CRC
# =====================================================================

GENERATORS_POLY = {"1011": "x³ + x + 1", "1101": "x³ + x² + 1", "10011": "x⁴ + x + 1",
                   "11001": "x⁴ + x³ + 1", "100101": "x⁵ + x² + 1"}


def crc_divide(bits: str, gen: str) -> tuple[str, list[str]]:
    """Mod-2 long division of `bits` by `gen`. Returns (remainder, step lines)."""
    r = len(gen) - 1
    work = list(bits)
    steps = []
    for i in range(len(bits) - r):
        if work[i] == "1":
            before = "".join(work[i:i + r + 1])
            for j in range(r + 1):
                work[i + j] = "0" if work[i + j] == gen[j] else "1"
            after = "".join(work[i:i + r + 1])
            steps.append(f"{before} XOR {gen} = {after}")
    return "".join(work[-r:]), steps


def gen_crc_sender(rng: random.Random, split: str = "train") -> Problem:
    gen = rng.choice(list(GENERATORS_POLY))
    n = rng.randint(6, 10) if split == "train" else rng.randint(10, 14)
    data = "1" + "".join(rng.choice("01") for _ in range(n - 1))
    r = len(gen) - 1
    rem, steps = crc_divide(data + "0" * r, gen)
    frame = data + rem
    q = rng.choice([
        f"Data bits {data} are to be sent using CRC with generator polynomial G(x) = {GENERATORS_POLY[gen]} "
        f"({gen}). Find the CRC remainder and the transmitted frame.",
        f"Using the CRC generator {gen}, compute the check bits for the message {data} and give the bit string "
        "that is actually transmitted.",
    ])
    sol = (
        f"Generator {gen} has degree r = {r}, so append {r} zeros: {data} → {data + '0' * r}.\n\n"
        "Divide by the generator using modulo-2 (XOR) division. At each step where the leading bit is 1, XOR "
        "with the generator:\n\n" + "\n".join(f"- {s}" for s in steps)
        + f"\n\nRemainder (CRC) = **{rem}**.\n\nTransmitted frame = data + CRC = {data} + {rem} = **{frame}**.\n\n"
        f"**Final answer:** CRC remainder = {rem}, transmitted frame = {frame}"
    )
    return Problem("CN", "crc/sender", q, sol, {"type": "tokens", "values": [rem, frame]})


def gen_crc_receiver(rng: random.Random, split: str = "train") -> Problem:
    gen = rng.choice(list(GENERATORS_POLY))
    n = rng.randint(6, 10) if split == "train" else rng.randint(10, 14)
    data = "1" + "".join(rng.choice("01") for _ in range(n - 1))
    r = len(gen) - 1
    frame = data + crc_divide(data + "0" * r, gen)[0]
    if rng.random() < 0.5:
        i = rng.randrange(len(frame))
        frame = frame[:i] + ("1" if frame[i] == "0" else "0") + frame[i + 1:]
    rem, steps = crc_divide(frame, gen)
    error = "1" in rem
    q = (f"A receiver gets the frame {frame}. The generator is {gen} ({GENERATORS_POLY[gen]}). "
         "Compute the remainder at the receiver and state whether an error is detected.")
    sol = (
        f"The receiver divides the whole received frame by {gen} (modulo-2):\n\n"
        + "\n".join(f"- {s}" for s in steps)
        + f"\n\nRemainder = **{rem}**. "
        + ("It is non-zero, so the receiver **detects an error** and discards the frame."
           if error else "It is all zeros, so **no error is detected** and the frame is accepted.")
        + f"\n\n**Final answer:** Remainder = {rem}; " + ("error detected" if error else "no error detected")
    )
    return Problem("CN", "crc/receiver", q, sol, {"type": "tokens", "values": [rem]})


# =====================================================================
# Internet checksum
# =====================================================================

def ones_complement_sum(words: list[int]) -> tuple[int, list[str]]:
    total, steps = 0, []
    for w in words:
        s = total + w
        if s > 0xFFFF:
            steps.append(f"{total:04X} + {w:04X} = {s:05X} → wrap carry → {(s & 0xFFFF) + 1:04X}")
            s = (s & 0xFFFF) + 1
        else:
            steps.append(f"{total:04X} + {w:04X} = {s:04X}")
        total = s
    return total, steps


def gen_checksum(rng: random.Random, split: str = "train") -> Problem:
    k = rng.randint(3, 5) if split == "train" else rng.randint(5, 7)
    words = [rng.randint(0, 0xFFFF) for _ in range(k)]
    total, steps = ones_complement_sum(words)
    cks = (~total) & 0xFFFF
    q = (f"Compute the 16-bit Internet checksum (one's complement of the one's complement sum) of the following "
         f"16-bit words given in hexadecimal: {', '.join(f'{w:04X}' for w in words)}.")
    sol = (
        "Add the words using one's complement arithmetic (a carry out of bit 15 is wrapped around and added "
        f"back):\n\n- Start with {words[0]:04X}\n" + "\n".join(f"- {s}" for s in steps[1:]) +
        f"\n\nSum = {total:04X}. Checksum = one's complement of the sum = FFFF − {total:04X} = **{cks:04X}**.\n\n"
        f"**Final answer:** Checksum = 0x{cks:04X}"
    )
    return Problem("CN", "checksum/internet", q, sol, {"type": "tokens", "values": [f"{cks:04X}"]})


# =====================================================================
# Delays
# =====================================================================

def _ms(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def gen_delay(rng: random.Random, split: str = "train") -> Problem:
    # The test split uses disjoint parameter values so held-out problems are never seen in training.
    if split == "train":
        L = rng.choice([500, 1000, 1250, 1500, 2000, 4000, 8000])       # bytes
        R = rng.choice([1, 2, 4, 5, 10, 20, 50, 100])                    # Mbps
        d = rng.choice([50, 100, 200, 400, 500, 1000, 2000, 3000, 5000]) # km
    else:
        L = rng.choice([600, 750, 3000, 6000, 12000])
        R = rng.choice([3, 8, 25, 40, 200])
        d = rng.choice([150, 250, 750, 1500, 4000])
    links = 1 if rng.random() < 0.6 else rng.randint(2, 4)
    t_tx = 8 * L / (R * 1e3)      # ms
    t_pr = d / 200                # ms, speed 2e8 m/s
    total = links * (t_tx + t_pr)
    path = "a link" if links == 1 else f"a path of {links} identical store-and-forward links (each"
    q = (f"A packet of {L} bytes is sent over {path} with bandwidth {R} Mbps and length {d} km"
         + (")" if links > 1 else "") + ". Signals propagate at 2 × 10^8 m/s. Ignoring processing and queuing "
         "delays, find the transmission delay, the propagation delay"
         + (" per link" if links > 1 else "") + " and the total end-to-end delay, in milliseconds.")
    sol = (
        f"Transmission delay = L / R = ({L} × 8 bits) / ({R} × 10^6 bps) = {8 * L} / {R * 10**6} s = "
        f"**{_ms(t_tx)} ms**\n\n"
        f"Propagation delay = d / s = ({d} × 10^3 m) / (2 × 10^8 m/s) = **{_ms(t_pr)} ms**\n\n"
        + (f"Total = {_ms(t_tx)} + {_ms(t_pr)} = **{_ms(total)} ms**" if links == 1 else
           f"With store-and-forward, every link adds one transmission and one propagation delay: "
           f"total = {links} × ({_ms(t_tx)} + {_ms(t_pr)}) = **{_ms(total)} ms**")
        + f"\n\n**Final answer:** Transmission delay = {_ms(t_tx)} ms, propagation delay = {_ms(t_pr)} ms, "
        f"total delay = {_ms(total)} ms"
    )
    return Problem("CN", "delay/" + ("single" if links == 1 else "store_forward"), q, sol,
                   {"type": "numbers", "values": [round(t_tx, 3), round(t_pr, 3), round(total, 3)],
                    "rel_tol": 0.01})


# =====================================================================
# Link utilisation (flow control)
# =====================================================================

def gen_window(rng: random.Random, split: str = "train") -> Problem:
    if split == "train":  # disjoint values for the test split, as in gen_delay
        L = rng.choice([1000, 1500, 2000, 4000, 8000])   # bytes
        R = rng.choice([1, 2, 4, 5, 10])                   # Mbps
        t_pr = rng.choice([2, 5, 10, 20, 25, 40, 50])      # ms
    else:
        L = rng.choice([1200, 2500, 5000])
        R = rng.choice([3, 8, 20])
        t_pr = rng.choice([3, 15, 30, 60])
    t_tx = 8 * L / (R * 1e3)                               # ms
    a = t_pr / t_tx
    kind = rng.choice(["sw", "gbn", "sr", "minw"])
    w = rng.choice([3, 4, 7, 8, 15, 16, 31]) if kind in ("gbn", "sr") else 1
    base = (f"Frames are {L} bytes long, the link bandwidth is {R} Mbps and the one-way propagation delay is "
            f"{t_pr} ms. Ignore ACK transmission time and processing delays.")
    setup = (f"Transmission time T_t = ({L} × 8) / ({R} × 10^6) s = {_ms(t_tx)} ms. "
             f"a = T_p / T_t = {t_pr} / {_ms(t_tx)} = {a:.4g}.\n\n")
    if kind == "minw":
        need = 1 + 2 * a
        wmin = math.ceil(need - 1e-9)
        bits_gbn = math.ceil(math.log2(wmin + 1))
        bits_sr = math.ceil(math.log2(2 * wmin))
        q = base + (" What is the minimum sender window size (in frames) needed to keep the link fully utilised, "
                    "and how many sequence-number bits does that window need with Go-Back-N and with Selective Repeat?")
        sol = (setup + f"Full utilisation needs W ≥ 1 + 2a = {need:.4g}, so W = **{wmin}** frames.\n\n"
               f"Go-Back-N needs W ≤ 2^k − 1 → k = ⌈log2({wmin} + 1)⌉ = **{bits_gbn}** bits.\n\n"
               f"Selective Repeat needs W ≤ 2^(k−1) → k = ⌈log2(2 × {wmin})⌉ = **{bits_sr}** bits.\n\n"
               f"**Final answer:** Minimum window = {wmin} frames, {bits_gbn} sequence bits (Go-Back-N), "
               f"{bits_sr} sequence bits (Selective Repeat)")
        return Problem("CN", "flow_control/min_window", q, sol,
                       {"type": "numbers", "values": [wmin, bits_gbn, bits_sr]})
    eff = min(1.0, w / (1 + 2 * a))
    thr = eff * R
    name = {"sw": "stop-and-wait", "gbn": f"Go-Back-N with window size {w}", "sr": f"Selective Repeat with window size {w}"}[kind]
    q = base + f" Using {name}, what is the link utilisation (efficiency) and the effective throughput?"
    formula = ("Stop-and-wait efficiency = 1 / (1 + 2a)" if kind == "sw" else
               f"Sliding-window efficiency = min(1, W / (1 + 2a)) with W = {w}")
    sol = (setup + f"{formula} = {100 * eff:.2f}%.\n\n"
           f"Throughput = efficiency × bandwidth = {eff:.4f} × {R} Mbps = **{thr:.3f} Mbps**.\n\n"
           f"**Final answer:** Efficiency = {100 * eff:.2f}%, throughput = {thr:.3f} Mbps")
    return Problem("CN", f"flow_control/{kind}", q, sol,
                   {"type": "numbers", "values": [round(100 * eff, 2), round(thr, 3)], "rel_tol": 0.01})


# =====================================================================
# Driver
# =====================================================================

GENERATORS = [(gen_subnet_analyse, 0.22), (gen_subnet_divide, 0.13), (gen_crc_sender, 0.15),
              (gen_crc_receiver, 0.08), (gen_checksum, 0.10), (gen_delay, 0.14), (gen_window, 0.18)]


def generate(n: int, seed: int, split: str = "train") -> list[dict]:
    return run_generators(GENERATORS, "cn", n, seed, split)


if __name__ == "__main__":
    cli(GENERATORS, "cn", "CN", default_n=900)
