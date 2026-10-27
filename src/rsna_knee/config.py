"""Shared constants."""

# Order must match sample_submission.csv (columns 1..12).
TARGET_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# One series per plane is stacked into the 3 input channels, in this order.
PLANES = ["Sagittal", "Coronal", "Axial"]

DEFAULT_IMG_SIZE = 224
