# Facial Recognition Model

A privacy-first Python CLI that enrolls consenting people from reference photos
and identifies them in group images. Detection and recognition happen locally;
the tool does not upload photos or embeddings.

> [!IMPORTANT]
> This project is for consent-based personal and educational use. Do not use it
> for surveillance or decisions affecting access, employment, housing, credit,
> healthcare, education, insurance, or law enforcement.

## How it works

1. YuNet detects each face and its alignment landmarks.
2. SFace converts each aligned face into a normalized numeric embedding.
3. Enrollment averages several embeddings for each person.
4. Recognition compares detected faces with enrolled identities using cosine
   similarity.
5. Matches above the threshold receive a label; other faces remain `Unknown`.

The neural networks are pretrained. You do not need to train a model from
scratch or own a GPU.

## Requirements

- Python 3.11 or newer
- macOS, Linux, or Windows
- Internet access once, to download the pinned OpenCV model files
- Clear reference photos used with every person's consent

## Installation

```bash
git clone https://github.com/fayadhkader-design/Facial-Recognition-Model.git
cd Facial-Recognition-Model
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Download checksum-verified YuNet and SFace models:

```bash
face-recognition download-models
```

The files are stored in `./models` by default and are ignored by Git. Use
`--model-directory PATH` or the `FACE_RECOGNITION_MODEL_DIR` environment
variable to choose another location.

## Enroll people

Create one directory per identity and add three to five varied photos of that
person. Every reference image must contain exactly one detectable face.

```text
references/
├── Alice/
│   ├── front.jpg
│   ├── outdoors.jpg
│   └── side-angle.png
└── Bob/
    ├── photo-1.jpg
    └── photo-2.jpg
```

Then build the local embedding database:

```bash
face-recognition enroll \
  --references ./references \
  --database ./data/faces.npz
```

Reference photos and `.npz` databases are ignored by Git. The database contains
biometric information and should still be protected like the original photos.

## Recognize a group photo

```bash
face-recognition recognize \
  --image ./group.jpg \
  --database ./data/faces.npz \
  --output ./outputs/group-labeled.jpg
```

The input is never modified. The command writes the annotated output and prints
a JSON summary:

```json
{
  "face_count": 2,
  "faces": [
    {
      "box": {"height": 140, "width": 120, "x": 88, "y": 42},
      "known": true,
      "label": "Alice",
      "similarity": 0.721834
    }
  ],
  "input": "group.jpg",
  "output": "outputs/group-labeled.jpg",
  "status": "ok"
}
```

## Choosing a threshold

The default cosine threshold is `0.363`, based on OpenCV's published SFace
guidance. Real collections differ, so calibrate it using photos that were not
used for enrollment:

```bash
face-recognition recognize ... --threshold 0.45
```

- Raise the threshold to reduce false matches.
- Lower it to reduce missed matches.
- Prefer `Unknown` over assigning the wrong identity.

Do not treat a similarity score as proof of identity.

## Limitations

Accuracy can drop with small or blurry faces, masks, harsh lighting, extreme
angles, aging, or major appearance changes. Performance can also differ across
demographic groups and camera conditions. Always keep a human in the loop and
do not use this tool in high-stakes settings.

This version supports still images only. It does not process video, webcams, or
remote URLs.

## Troubleshooting

- **Model not found:** run `face-recognition download-models`, or pass the same
  `--model-directory` to every command.
- **Zero faces in a reference:** use a clearer, larger, front-facing photo.
- **Multiple faces in a reference:** crop it so only the intended person remains.
- **Too many `Unknown` results:** add varied enrollment photos, then cautiously
  test a slightly lower threshold.
- **Wrong names:** raise the threshold and replace ambiguous reference photos.
- **Cannot read an image:** convert HEIC or unsupported formats to JPG or PNG.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy src
pytest --cov=face_recognition
```

Tests use generated arrays and mocked model boundaries. No real face images or
biometric embeddings are stored in this repository.

## License

[MIT](LICENSE). OpenCV and individual OpenCV Zoo models retain their respective
licenses.
