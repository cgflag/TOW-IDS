#!/usr/bin/env python
"""Preprocess training traffic into N x M image-like groups.

This version follows the paper's intent more closely:
- For each target size M in {32, 60, 116, 228, 452}, build an independent dataset.
- Use consecutive packets (preserve temporal order).
- Group label rule: if one or more attack packets appear in a group, label it Abnormal.
"""

import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scapy.all import Raw, rdpcap
from tqdm import tqdm


TARGET_SIZES = [32, 60, 116, 228, 452]
PCAP_PATH = "Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap"
LABEL_PATH = "y_train.csv"
LABELS_OUTPUT_DIR = "labels"
NORMALIZED_OUTPUT_DIR = "normalized_packet_data"


def extract_packet_payload(packet) -> List[int]:
    payload = packet[Raw].load if Raw in packet else b""
    return list(payload)


def adjust_packet_size(payload_values: List[int], target_size: int) -> List[int]:
    if len(payload_values) >= target_size:
        return payload_values[:target_size]
    return payload_values + [0] * (target_size - len(payload_values))


def group_consecutive_packets(
    packets_for_m: List[List[int]],
    packet_labels: List[str],
    n_size: int,
) -> Tuple[np.ndarray, List[str]]:
    grouped_packets = []
    grouped_labels = []

    cursor = 0
    total_packets = len(packets_for_m)
    zero_packet = [0] * n_size

    while cursor < total_packets:
        group_packets = packets_for_m[cursor : cursor + n_size]
        group_packet_labels = packet_labels[cursor : cursor + n_size]
        cursor += n_size

        if len(group_packets) < n_size:
            # Keep final group shape stable with zero-padding packets.
            group_packets += [zero_packet] * (n_size - len(group_packets))

        group_label = "Abnormal" if "Abnormal" in group_packet_labels else "Normal"
        grouped_packets.append(group_packets)
        grouped_labels.append(group_label)

    return np.array(grouped_packets, dtype=np.float32), grouped_labels


def save_group_labels(labels: List[str], size_n: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{size_n}x{size_n}.csv")
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Group Index", "Label"])
        for idx, label in enumerate(labels):
            writer.writerow([idx, label])


def save_normalized_groups(groups: np.ndarray, size_n: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    normalized = groups / 255.0
    output_path = os.path.join(output_dir, f"{size_n}x{size_n}_normalized.npy")
    np.save(output_path, normalized)


def main() -> None:
    packets = rdpcap(PCAP_PATH)
    labels_df = pd.read_csv(LABEL_PATH, header=None, names=["packet_id", "target", "label_of_protocol"])
    packet_labels = labels_df["target"].astype(str).tolist()

    if len(packet_labels) != len(packets):
        raise ValueError(
            f"Label count ({len(packet_labels)}) does not match packet count ({len(packets)})."
        )

    print(len(packets))

    # Build raw payload cache once to avoid repeated packet parsing per target size.
    payload_cache = []
    for packet in tqdm(packets, total=len(packets)):
        payload_cache.append(extract_packet_payload(packet))

    processed_data: Dict[Tuple[int, int], np.ndarray] = {}

    for size_n in TARGET_SIZES:
        packets_for_size = [adjust_packet_size(payload, size_n) for payload in payload_cache]
        groups, group_labels = group_consecutive_packets(packets_for_size, packet_labels, size_n)

        abnormal_count = sum(1 for item in group_labels if item == "Abnormal")
        normal_count = len(group_labels) - abnormal_count
        print(f"Key: ({size_n}, {size_n})")
        print(f"  Label 'Normal' has {normal_count} groups")
        print(f"  Label 'Abnormal' has {abnormal_count} groups")

        processed_data[(size_n, size_n)] = groups
        save_group_labels(group_labels, size_n, LABELS_OUTPUT_DIR)
        save_normalized_groups(groups, size_n, NORMALIZED_OUTPUT_DIR)

    for key, groups in processed_data.items():
        print(f"Size {key}: Number of packets = {len(groups)}")


if __name__ == "__main__":
    main()
