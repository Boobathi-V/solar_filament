import json

with open("train_and_predict_sota_fixed.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell_2 = [
    "# ============================================================\n",
    "# 2. MOUNT GOOGLE DRIVE & AUTO-DISCOVER DATASET\n",
    "# ============================================================\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount(\"/content/drive\")\n",
    "    print(\"✅ Google Drive mounted successfully.\")\n",
    "except Exception:\n",
    "    print(\"ℹ️ Google Drive mount skipped or already active.\")\n",
    "\n",
    "POSSIBLE_DIRS = [\n",
    "    \"/content/drive/MyDrive/kaggle_SF/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\",\n",
    "    \"/content/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\",\n",
    "    \"/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\",\n",
    "    \"/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\",\n",
    "    \"filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\"\n",
    "]\n",
    "\n",
    "DATA_BASE = None\n",
    "for p in POSSIBLE_DIRS:\n",
    "    if os.path.exists(p) and os.path.exists(os.path.join(p, \"train\")):\n",
    "        DATA_BASE = p\n",
    "        break\n",
    "\n",
    "if DATA_BASE is None:\n",
    "    print(\"🔍 Searching for dataset files across Google Drive / Colab...\")\n",
    "    for root in [\"/content/drive/Othercomputers\", \"/content/drive/MyDrive\", \"/content/drive\", \"/content\", \".\", \"/kaggle/input\"]:\n",
    "        if os.path.exists(root):\n",
    "            hits = list(Path(root).rglob(\"MAGFiLO_1.0_Annotations_kaggle2026_train.json\"))\n",
    "            if hits:\n",
    "                DATA_BASE = str(hits[0].parent.parent)\n",
    "                print(f\"✅ Auto-discovered dataset at: {DATA_BASE}\")\n",
    "                break\n",
    "\n",
    "if DATA_BASE is None or not os.path.exists(DATA_BASE):\n",
    "    raise FileNotFoundError(\"MAGFiLO dataset folder not found! Please check Drive path.\")\n",
    "\n",
    "print(f\"✅ Using Dataset Path: {DATA_BASE}\")\n"
]

nb["cells"][2]["source"] = new_cell_2

with open("train_and_predict_sota_fixed.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("SUCCESS: Updated train_and_predict_sota_fixed.ipynb")
