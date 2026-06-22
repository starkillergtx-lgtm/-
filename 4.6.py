# ============================================================
# Раздел 4.6
# Сценарная расчётная проверка выбора уровней при изменении
# пропускной способности канала
#
# Терминология для диплома:
# - не "ABR-эксперимент";
# - "расчётная проверка";
# - "выбор уровня при изменении пропускной способности";
# - "сценарный анализ".
#
# Что делает скрипт:
# 1. Использует полную сетку уровней YouTube и предложенной лестницы.
# 2. Если есть channel_ladder_bitrates.csv, НЕ кодирует видео заново.
# 3. Если CSV нет, кодирует reference_1.mp4 в представления всех уровней.
# 4. Строит несколько профилей пропускной способности:
#    - высокий канал -> падение -> восстановление;
#    - плавное падение/восстановление;
#    - повторные колебания канала.
#    Масштабы профилей охватывают 4K, 1440p/верхне-средний,
#    1080p, 720p и 480p области.
# 5. Для каждого профиля сравнивает две лестницы:
#    - средний MOS;
#    - средний выбранный битрейт;
#    - общий объём данных;
#    - число переключений;
#    - эффективность MOS/Мбит/с.
#
# Ожидаемый вход:
#   Windows: C:\video_exp\originals\reference_1.mp4
#   Colab/Linux: /content/video_exp/originals/reference_1.mp4
#
# Основная папка результатов:
#   Windows: C:\video_exp\channel_ladder_test_4_6_uhd_full90
#   Colab/Linux: /content/video_exp/channel_ladder_test_4_6_uhd_full90
#
# Если в этой папке уже есть channel_ladder_bitrates.csv,
# скрипт сразу переходит к сценариям и НЕ перекодирует видео.
# ============================================================

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. ПАРАМЕТРЫ
# ============================================================

INPUT_DIR = Path(r"C:\video_exp\originals")

# ВАЖНО: новая UHD-нормировка меняет размеры proposed-уровней,
# поэтому используется новая папка результатов. Старый channel_ladder_bitrates.csv
# от версии 4000×2109 нельзя переиспользовать без перекодирования.
OUTPUT_DIR = Path(r"C:\video_exp\channel_ladder_test_4_6_uhd_full90")

FFMPEG_PATH = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
FFPROBE_PATH = Path(r"C:\ffmpeg\bin\ffprobe.exe")

REFERENCE_FILES = [
    "reference_1.mp4",
]

CODEC = "libx264"
CRF = 23
PRESET = "veryfast"
PIX_FMT = "yuv420p"

# 90 секунд - компромисс между устойчивостью оценки и временем кодирования.
ENCODE_SAMPLE_START_SEC = 0
ENCODE_SAMPLE_DURATION_SEC = 90

SKIP_EXISTING_ENCODED = True
LOAD_EXISTING_BITRATES_IF_AVAILABLE = True

INTERVAL_DURATION_SEC = 5
SIMULATION_DURATION_SEC = 90
SAFETY_FACTOR = 0.85

# Отношение low/high по форме профиля из работы Bampis:
# 100 / 250 = 0.4.
DEFAULT_LOW_TO_HIGH_RATIO = 0.40

CREATE_DEMO_VIDEOS = False
DEMO_REFERENCE_FILE = "reference_1.mp4"

# Автоматическая поправка путей под среду запуска.
# Если скрипт запущен в Google Colab / Linux, Windows-папки вида C:\video_exp
# недоступны. Поэтому используем /content/video_exp и системные ffmpeg/ffprobe.
IS_WINDOWS = (os.name == "nt")

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if not IS_WINDOWS:
    INPUT_DIR = Path("/content/video_exp/originals")
    OUTPUT_DIR = Path("/content/video_exp/channel_ladder_test_4_6_uhd_full90")
    FFMPEG_PATH = Path("ffmpeg")
    FFPROBE_PATH = Path("ffprobe")



# ============================================================
# 1. ЛЕСТНИЦЫ
# ============================================================

PROPOSED_LADDER = [
    # Новая компромиссная логика:
    # исходная авторская шкала 4096×2160 приведена к UHD-шкале 3840×2160
    # по ширине: width_norm = width_raw × 3840 / 4096.
    # Высота остаётся той же, как в новых графиках 1–4.
    {"level_id": "P1", "mos": 5.0, "width": 3750, "height": 2109, "label": "3750x2109"},
    {"level_id": "P2", "mos": 4.5, "width": 1957, "height": 1100, "label": "1957x1100"},
    {"level_id": "P3", "mos": 4.0, "width": 980,  "height": 551,  "label": "980x551"},
    {"level_id": "P4", "mos": 3.5, "width": 572,  "height": 322,  "label": "572x322"},
    {"level_id": "P5", "mos": 3.0, "width": 484,  "height": 272,  "label": "484x272"},
    {"level_id": "P6", "mos": 2.0, "width": 275,  "height": 154,  "label": "275x154"},
]

YOUTUBE_LADDER = [
    {"level_id": "Y1", "width": 3840, "height": 2160, "label": "2160p / 3840x2160"},
    {"level_id": "Y2", "width": 2560, "height": 1440, "label": "1440p / 2560x1440"},
    {"level_id": "Y3", "width": 1920, "height": 1080, "label": "1080p / 1920x1080"},
    {"level_id": "Y4", "width": 1280, "height": 720,  "label": "720p / 1280x720"},
    {"level_id": "Y5", "width": 854,  "height": 480,  "label": "480p / 854x480"},
    {"level_id": "Y6", "width": 640,  "height": 360,  "label": "360p / 640x360"},
    {"level_id": "Y7", "width": 426,  "height": 240,  "label": "240p / 426x240"},
]


# ============================================================
# 2. Авторская зависимость разрешение(MOS)
#    Полином Лагранжа 3-й степени по 4 опорным точкам
# ============================================================

MOS_BASE = np.array([5.0, 4.0, 3.0, 2.0])
WIDTH_BASE = np.array([3750.0, 980.0, 484.0, 275.0])


def lagrange_eval(x_points, y_points, x_eval):
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    y_eval = np.zeros_like(x_eval, dtype=float)

    for i in range(len(x_points)):
        basis = np.ones_like(x_eval, dtype=float)
        for j in range(len(x_points)):
            if i != j:
                basis *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        y_eval += y_points[i] * basis

    return y_eval


def width_to_mos(width):
    """
    Обратное отображение по авторской зависимости разрешение(MOS).

    Предложенная лестница имеет MOS напрямую.
    YouTube переносится на MOS через эту авторскую зависимость.
    """
    width = float(width)

    if width <= WIDTH_BASE.min():
        return float(MOS_BASE.min())
    if width >= WIDTH_BASE.max():
        return float(MOS_BASE.max())

    lo, hi = 2.0, 5.0

    for _ in range(80):
        mid = (lo + hi) / 2.0
        w_mid = float(lagrange_eval(MOS_BASE, WIDTH_BASE, np.array([mid]))[0])

        if w_mid < width:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def add_mos_to_ladders():
    for item in PROPOSED_LADDER:
        item["ladder"] = "proposed"
        item["mos"] = float(item["mos"])

    for item in YOUTUBE_LADDER:
        item["ladder"] = "youtube"
        item["mos"] = float(width_to_mos(item["width"]))


def all_ladders():
    return {
        "proposed": PROPOSED_LADDER,
        "youtube": YOUTUBE_LADDER,
    }


# ============================================================
# 3. Сценарии проверки
# ============================================================

# Все сценарии используют одну форму:
# высокий канал -> падение -> восстановление.
#
# Различается масштаб профиля. Это не читерство, а сценарный анализ:
# проверяется, как лестницы ведут себя при разных уровнях доступной полосы.
#
# high_calibration_pairs:
# - список уровней, относительно которых выбирается "высокий" канал;
# - high = max(битрейт этих уровней) / SAFETY_FACTOR * high_margin.
#
# low_to_high_ratio:
# - отношение нижнего состояния канала к высокому;
# - 0.40 соответствует форме профиля Bampis: 100/250.
#
# profile_shape:
# - "step": резкое падение и восстановление;
# - "gradual": плавное падение и плавное восстановление.
SCENARIOS = [
    {
        "scenario_id": "S0_high_4k_step",
        "scenario_name": "Высокий канал 4K/исходный уровень: резкое падение",
        "description": "Высокий канал откалиброван по proposed P1 и YouTube Y1. Проверяется поведение лестниц в верхней области качества, включая около-UHD уровни.",
        "high_calibration_pairs": [("proposed", "P1"), ("youtube", "Y1")],
        "high_margin": 1.03,
        "low_to_high_ratio": 0.40,
        "profile_shape": "step",
    },
    {
        "scenario_id": "S1_baseline_1080_step",
        "scenario_name": "Базовый сценарий 1080p/P2: резкое падение",
        "description": "Высокий канал откалиброван по proposed P2 и YouTube Y3. Это базовая проверка верхне-средней области качества.",
        "high_calibration_pairs": [("proposed", "P2"), ("youtube", "Y3")],
        "high_margin": 1.10,
        "low_to_high_ratio": 0.40,
        "profile_shape": "step",
    },
    {
        "scenario_id": "S2_upper_mid_1440_step",
        "scenario_name": "Верхне-средний канал 1440p/P2: резкое падение",
        "description": "Высокий канал откалиброван по proposed P2 и YouTube Y2. Проверяется возможность экономии данных при сопоставимом MOS.",
        "high_calibration_pairs": [("proposed", "P2"), ("youtube", "Y2")],
        "high_margin": 1.03,
        "low_to_high_ratio": 0.40,
        "profile_shape": "step",
    },
    {
        "scenario_id": "S3_medium_720_step",
        "scenario_name": "Средний канал 720p/P3: резкое падение",
        "description": "Высокий канал откалиброван по proposed P3 и YouTube Y4. Проверяется поведение в средней области качества.",
        "high_calibration_pairs": [("proposed", "P3"), ("youtube", "Y4")],
        "high_margin": 1.03,
        "low_to_high_ratio": 0.40,
        "profile_shape": "step",
    },
    {
        "scenario_id": "S4_low_480_step",
        "scenario_name": "Пониженный канал 480p/P4: резкое падение",
        "description": "Высокий канал откалиброван по proposed P4 и YouTube Y5. Проверяется поведение в нижне-средней области качества.",
        "high_calibration_pairs": [("proposed", "P4"), ("youtube", "Y5")],
        "high_margin": 1.03,
        "low_to_high_ratio": 0.40,
        "profile_shape": "step",
    },
    {
        "scenario_id": "S5_high_4k_gradual",
        "scenario_name": "Высокий канал 4K/исходный уровень: плавное изменение",
        "description": "Масштаб S0, но падение и восстановление происходят плавно.",
        "high_calibration_pairs": [("proposed", "P1"), ("youtube", "Y1")],
        "high_margin": 1.03,
        "low_to_high_ratio": 0.40,
        "profile_shape": "gradual",
    },
    {
        "scenario_id": "S6_upper_mid_1440_gradual",
        "scenario_name": "Верхне-средний канал 1440p/P2: плавное изменение",
        "description": "Масштаб S2, но падение и восстановление происходят плавно.",
        "high_calibration_pairs": [("proposed", "P2"), ("youtube", "Y2")],
        "high_margin": 1.03,
        "low_to_high_ratio": 0.40,
        "profile_shape": "gradual",
    },
    {
        "scenario_id": "S7_upper_mid_1440_oscillating",
        "scenario_name": "Верхне-средний канал 1440p/P2: повторные колебания",
        "description": "Масштаб S2, но канал дважды падает и восстанавливается. Проверяется устойчивость выбора уровней при повторных изменениях.",
        "high_calibration_pairs": [("proposed", "P2"), ("youtube", "Y2")],
        "high_margin": 1.03,
        "low_to_high_ratio": 0.40,
        "profile_shape": "oscillating",
    },
]


# ============================================================
# 4. Служебные функции
# ============================================================

def resolve_tool(path_obj, fallback_name):
    if Path(path_obj).exists():
        return str(path_obj)

    from shutil import which
    found = which(fallback_name)
    if found:
        return found

    return None


def make_even(value):
    value = int(round(value))
    if value < 2:
        return 2
    return value if value % 2 == 0 else value - 1


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "encoded_representations_4_6").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "encoded_representations_4_6" / "proposed").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "encoded_representations_4_6" / "youtube").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures_scenarios").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "demo").mkdir(parents=True, exist_ok=True)


def check_tools():
    global FFMPEG_PATH, FFPROBE_PATH

    ffmpeg_resolved = resolve_tool(FFMPEG_PATH, "ffmpeg")
    ffprobe_resolved = resolve_tool(FFPROBE_PATH, "ffprobe")

    if not ffmpeg_resolved:
        print("ОШИБКА: не найден ffmpeg.")
        print(f"Текущий FFMPEG_PATH: {FFMPEG_PATH}")
        sys.exit(1)

    if not ffprobe_resolved:
        print("ОШИБКА: не найден ffprobe.")
        print(f"Текущий FFPROBE_PATH: {FFPROBE_PATH}")
        sys.exit(1)

    FFMPEG_PATH = ffmpeg_resolved
    FFPROBE_PATH = ffprobe_resolved

    result = subprocess.run(
        [FFMPEG_PATH, "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        print("ОШИБКА: FFmpeg не запускается.")
        sys.exit(1)

    print(result.stdout.splitlines()[0])


def run_cmd(cmd, quiet=False):
    if not quiet:
        print(" ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Команда завершилась с ошибкой: {result.returncode}")

    return result


def ffprobe_json(path):
    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]
    result = run_cmd(cmd, quiet=True)
    return json.loads(result.stdout)


def get_video_info(path):
    data = ffprobe_json(path)
    video_stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
    duration = float(data["format"]["duration"])

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "duration": duration,
        "fps": video_stream.get("r_frame_rate", "unknown"),
        "size_bytes": int(data["format"].get("size", os.path.getsize(path)))
    }


def get_file_bitrate_mbps(path):
    info = get_video_info(path)
    size_bits = info["size_bytes"] * 8
    return size_bits / info["duration"] / 1_000_000.0


def maybe_upload_reference_file(expected_name):
    """
    Для Colab: если reference_1.mp4 не найден в /content/video_exp/originals,
    предлагаем загрузить файл через стандартное окно files.upload().
    В локальном Windows-режиме функция ничего не делает.
    """
    if not IN_COLAB:
        return None

    try:
        from google.colab import files
    except Exception:
        return None

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nФайл {expected_name} не найден в {INPUT_DIR}.")
    print("Скрипт запущен не в Windows, а в Colab/Linux, поэтому путь C:\\video_exp здесь недоступен.")
    print("Загрузите reference_1.mp4 через окно загрузки:")

    uploaded = files.upload()

    for fname, content in uploaded.items():
        out_path = INPUT_DIR / fname
        out_path.write_bytes(content)
        print(f"  ✓ Загружено: {out_path} ({len(content) / (1024 * 1024):.1f} МБ)")

    direct = INPUT_DIR / expected_name
    if direct.exists():
        return direct

    stem = Path(expected_name).stem
    candidates = sorted(INPUT_DIR.glob(stem + ".*"))
    candidates = [
        p for p in candidates
        if p.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]
    ]

    if candidates:
        return candidates[0]

    return None


def find_reference_files():
    files = []

    for name in REFERENCE_FILES:
        path = INPUT_DIR / name

        if not path.exists():
            stem = Path(name).stem
            candidates = sorted(INPUT_DIR.glob(stem + ".*"))
            candidates = [
                p for p in candidates
                if p.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]
            ]

            if candidates:
                path = candidates[0]
            else:
                uploaded_path = maybe_upload_reference_file(name)
                if uploaded_path is not None and uploaded_path.exists():
                    path = uploaded_path
                else:
                    print(f"ОШИБКА: не найден файл: {INPUT_DIR / name}")
                    print(f"Также не найдено файлов вида: {stem}.*")
                    print("\nЧто сделать:")
                    print("  1) В Colab загрузите reference_1.mp4 в /content/video_exp/originals")
                    print("     или используйте появившееся окно files.upload().")
                    print("  2) В Windows положите файл в C:\\video_exp\\originals")
                    print("     или измените переменную INPUT_DIR в начале скрипта.")
                    sys.exit(1)

        files.append(path)

    print("\nНайдены исходные reference-видео:")
    for p in files:
        info = get_video_info(p)
        print(f"  {p.name}: {info['width']}x{info['height']}, {info['duration']:.1f} c, {info['fps']}")

    return files

def encoded_filename(reference_path, ladder_name, level):
    w_enc = make_even(level["width"])
    h_enc = make_even(level["height"])
    stem = reference_path.stem
    safe_level = level["level_id"]

    if ENCODE_SAMPLE_DURATION_SEC is None:
        sample_tag = "full"
    else:
        sample_tag = f"s{int(ENCODE_SAMPLE_START_SEC)}_d{int(ENCODE_SAMPLE_DURATION_SEC)}"

    return (
        OUTPUT_DIR
        / "encoded_representations_4_6"
        / ladder_name
        / f"{stem}_{sample_tag}_alllevels_{safe_level}_{w_enc}x{h_enc}_crf{CRF}_{PRESET}.mp4"
    )


# ============================================================
# 5. Кодирование представлений или загрузка готовых битрейтов
# ============================================================

def encode_video_level(reference_path, ladder_name, level):
    out_path = encoded_filename(reference_path, ladder_name, level)

    if out_path.exists() and SKIP_EXISTING_ENCODED:
        return out_path

    w_enc = make_even(level["width"])
    h_enc = make_even(level["height"])

    print(f"  Кодирование: {reference_path.name} -> {ladder_name} {level['level_id']} {w_enc}x{h_enc}")

    vf = f"scale={w_enc}:{h_enc}:flags=lanczos,setsar=1"

    cmd = [
        FFMPEG_PATH,
        "-y",
    ]

    if ENCODE_SAMPLE_START_SEC and ENCODE_SAMPLE_START_SEC > 0:
        cmd += ["-ss", str(ENCODE_SAMPLE_START_SEC)]

    cmd += ["-i", str(reference_path)]

    if ENCODE_SAMPLE_DURATION_SEC is not None:
        cmd += ["-t", str(ENCODE_SAMPLE_DURATION_SEC)]

    cmd += [
        "-vf", vf,
        "-c:v", CODEC,
        "-crf", str(CRF),
        "-preset", PRESET,
        "-pix_fmt", PIX_FMT,
        "-an",
        "-movflags", "+faststart",
        str(out_path)
    ]

    run_cmd(cmd, quiet=True)
    return out_path


def encode_all_representations(reference_files):
    rows = []

    print("\nКодирование представлений двух лестниц:")

    for ladder_name, ladder in all_ladders().items():
        print(f"\nЛестница: {ladder_name}")

        for level in ladder:
            for ref in reference_files:
                out_path = encode_video_level(ref, ladder_name, level)
                bitrate = get_file_bitrate_mbps(out_path)
                info = get_video_info(out_path)

                rows.append({
                    "ladder": ladder_name,
                    "reference": ref.name,
                    "level_id": level["level_id"],
                    "label": level["label"],
                    "mos": level["mos"],
                    "width_nominal": level["width"],
                    "height_nominal": level["height"],
                    "width_encoded": info["width"],
                    "height_encoded": info["height"],
                    "duration_sec": info["duration"],
                    "size_bytes": info["size_bytes"],
                    "bitrate_mbps": bitrate,
                    "file": str(out_path)
                })

                print(f"    {ref.name:15s} | {level['level_id']:>2s} | "
                      f"{info['width']}x{info['height']} | {bitrate:.3f} Mbps")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "encoded_files_detail.csv", index=False, encoding="utf-8-sig")

    summary = (
        df
        .groupby(
            [
                "ladder",
                "level_id",
                "label",
                "mos",
                "width_nominal",
                "height_nominal",
                "width_encoded",
                "height_encoded"
            ],
            as_index=False
        )
        .agg(
            avg_bitrate_mbps=("bitrate_mbps", "mean"),
            min_bitrate_mbps=("bitrate_mbps", "min"),
            max_bitrate_mbps=("bitrate_mbps", "max"),
            avg_size_mb=("size_bytes", lambda x: np.mean(x) / (1024 * 1024))
        )
    )

    summary = summary.sort_values(["ladder", "mos"], ascending=[True, False])
    summary.to_csv(OUTPUT_DIR / "channel_ladder_bitrates.csv", index=False, encoding="utf-8-sig")

    print(f"\nСохранено: {OUTPUT_DIR / 'encoded_files_detail.csv'}")
    print(f"Сохранено: {OUTPUT_DIR / 'channel_ladder_bitrates.csv'}")

    return summary


def validate_ladder_summary(df):
    required_cols = {
        "ladder", "level_id", "label", "mos",
        "width_nominal", "height_nominal", "avg_bitrate_mbps"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В channel_ladder_bitrates.csv нет колонок: {missing}")

    needed = [
        ("proposed", "P1"), ("proposed", "P2"), ("proposed", "P3"),
        ("proposed", "P4"), ("proposed", "P5"), ("proposed", "P6"),
        ("youtube", "Y1"), ("youtube", "Y2"), ("youtube", "Y3"),
        ("youtube", "Y4"), ("youtube", "Y5"), ("youtube", "Y6"), ("youtube", "Y7"),
    ]

    existing = set(zip(df["ladder"], df["level_id"]))

    absent = [pair for pair in needed if pair not in existing]
    if absent:
        raise ValueError(f"В channel_ladder_bitrates.csv нет уровней: {absent}")

    return df


def get_or_create_ladder_bitrates(reference_files):
    csv_path = OUTPUT_DIR / "channel_ladder_bitrates.csv"

    if LOAD_EXISTING_BITRATES_IF_AVAILABLE and csv_path.exists():
        print(f"\nНайден готовый файл битрейтов: {csv_path}")
        print("Кодирование представлений пропущено.")
        df = pd.read_csv(csv_path)
        return validate_ladder_summary(df)

    print("\nГотовый channel_ladder_bitrates.csv не найден. Запускаю кодирование.")
    return encode_all_representations(reference_files)


# ============================================================
# 6. Профили пропускной способности
# ============================================================

def get_level_bitrate(ladder_summary, ladder_name, level_id):
    row = ladder_summary[
        (ladder_summary["ladder"] == ladder_name) &
        (ladder_summary["level_id"] == level_id)
    ]

    if row.empty:
        raise ValueError(f"Не найден уровень {ladder_name} {level_id}")

    return float(row["avg_bitrate_mbps"].iloc[0])


def compute_high_bandwidth(ladder_summary, scenario):
    targets = []

    for ladder_name, level_id in scenario["high_calibration_pairs"]:
        targets.append(get_level_bitrate(ladder_summary, ladder_name, level_id))

    target = max(targets)
    return target / SAFETY_FACTOR * scenario["high_margin"]


def create_profile_for_scenario(ladder_summary, scenario, max_duration):
    high = compute_high_bandwidth(ladder_summary, scenario)
    low = high * scenario["low_to_high_ratio"]

    duration = min(
        SIMULATION_DURATION_SEC,
        int(max_duration // INTERVAL_DURATION_SEC) * INTERVAL_DURATION_SEC
    )

    times = np.arange(0, duration, INTERVAL_DURATION_SEC)
    n = len(times)

    rows = []
    shape = scenario["profile_shape"]

    for idx, t in enumerate(times):
        if shape == "step":
            # 1/3 high, 1/3 low, 1/3 high
            if idx < n / 3:
                bw = high
                phase = "high"
                phase_ru = "стабильный канал"
            elif idx < 2 * n / 3:
                bw = low
                phase = "drop"
                phase_ru = "падение канала"
            else:
                bw = high
                phase = "recovery"
                phase_ru = "восстановление"

        elif shape == "gradual":
            # high -> плавное падение -> low -> плавное восстановление
            pos = idx / max(n - 1, 1)

            if pos < 0.25:
                bw = high
                phase = "high"
                phase_ru = "стабильный канал"
            elif pos < 0.50:
                k = (pos - 0.25) / 0.25
                bw = high + (low - high) * k
                phase = "gradual_drop"
                phase_ru = "плавное падение"
            elif pos < 0.75:
                bw = low
                phase = "low"
                phase_ru = "низкий канал"
            else:
                k = (pos - 0.75) / 0.25
                bw = low + (high - low) * k
                phase = "gradual_recovery"
                phase_ru = "плавное восстановление"

        elif shape == "oscillating":
            # Повторные колебания: high -> low -> high -> low -> high.
            # Это не отдельная "модель Бамписа", а стресс-сценарий на той же
            # форме high/low, чтобы проверить устойчивость выбора уровней.
            pos = idx / max(n - 1, 1)

            if pos < 0.20:
                bw = high
                phase = "high_1"
                phase_ru = "стабильный канал"
            elif pos < 0.40:
                bw = low
                phase = "drop_1"
                phase_ru = "первое падение"
            elif pos < 0.60:
                bw = high
                phase = "recovery_1"
                phase_ru = "первое восстановление"
            elif pos < 0.80:
                bw = low
                phase = "drop_2"
                phase_ru = "второе падение"
            else:
                bw = high
                phase = "recovery_2"
                phase_ru = "второе восстановление"

        else:
            raise ValueError(f"Неизвестная форма профиля: {shape}")

        rows.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_name": scenario["scenario_name"],
            "interval_index": len(rows) + 1,
            "time_start_sec": int(t),
            "time_end_sec": int(t + INTERVAL_DURATION_SEC),
            "bandwidth_mbps": float(bw),
            "phase": phase,
            "phase_ru": phase_ru,
            "high_bandwidth_mbps": float(high),
            "low_bandwidth_mbps": float(low),
            "low_to_high_ratio": float(scenario["low_to_high_ratio"]),
            "profile_shape": shape,
            "scenario_description": scenario["description"]
        })

    return pd.DataFrame(rows)


def create_all_profiles(ladder_summary, max_duration):
    profiles = []

    for scenario in SCENARIOS:
        profiles.append(create_profile_for_scenario(ladder_summary, scenario, max_duration))

    df = pd.concat(profiles, ignore_index=True)
    df.to_csv(OUTPUT_DIR / "scenarios_channel_profiles.csv", index=False, encoding="utf-8-sig")

    print(f"\nСохранено: {OUTPUT_DIR / 'scenarios_channel_profiles.csv'}")

    profile_short = (
        df
        .groupby(["scenario_id", "scenario_name"], as_index=False)
        .agg(
            high_bandwidth_mbps=("high_bandwidth_mbps", "first"),
            low_bandwidth_mbps=("low_bandwidth_mbps", "first"),
            profile_shape=("profile_shape", "first"),
            intervals=("interval_index", "count")
        )
    )

    print("\nСценарии профиля:")
    print(profile_short.to_string(index=False))

    return df


# ============================================================
# 7. Расчёт выбора уровня
# ============================================================

def choose_level_for_bandwidth(ladder_df, bandwidth_mbps):
    allowed_bitrate = bandwidth_mbps * SAFETY_FACTOR

    # Максимальное качество, которое помещается в канал с запасом.
    candidates = ladder_df.sort_values("mos", ascending=False)
    fit = candidates[candidates["avg_bitrate_mbps"] <= allowed_bitrate]

    if len(fit) == 0:
        # Если даже нижний уровень не помещается, выбирается нижний уровень.
        return candidates.sort_values("mos", ascending=True).iloc[0]

    return fit.iloc[0]


def simulate_level_selection(ladder_summary, profiles):
    rows = []

    for scenario_id, scenario_profile in profiles.groupby("scenario_id", sort=False):
        for ladder_name in ["proposed", "youtube"]:
            ladder_df = ladder_summary[ladder_summary["ladder"] == ladder_name].copy()

            for _, interval in scenario_profile.iterrows():
                selected = choose_level_for_bandwidth(ladder_df, interval["bandwidth_mbps"])
                data_mb = selected["avg_bitrate_mbps"] * INTERVAL_DURATION_SEC / 8.0

                rows.append({
                    "scenario_id": scenario_id,
                    "scenario_name": interval["scenario_name"],
                    "ladder": ladder_name,
                    "interval_index": int(interval["interval_index"]),
                    "time_start_sec": int(interval["time_start_sec"]),
                    "time_end_sec": int(interval["time_end_sec"]),
                    "phase": interval["phase"],
                    "phase_ru": interval["phase_ru"],
                    "bandwidth_mbps": float(interval["bandwidth_mbps"]),
                    "allowed_bitrate_mbps": float(interval["bandwidth_mbps"] * SAFETY_FACTOR),
                    "selected_level_id": selected["level_id"],
                    "selected_label": selected["label"],
                    "selected_mos": float(selected["mos"]),
                    "selected_width": int(selected["width_nominal"]),
                    "selected_height": int(selected["height_nominal"]),
                    "selected_bitrate_mbps": float(selected["avg_bitrate_mbps"]),
                    "data_mb": float(data_mb)
                })

    timeline = pd.DataFrame(rows)
    timeline.to_csv(OUTPUT_DIR / "scenarios_ladder_selection_timeline.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for (scenario_id, ladder_name), g in timeline.groupby(["scenario_id", "ladder"], sort=False):
        level_changes = (g["selected_level_id"] != g["selected_level_id"].shift(1)).sum() - 1
        level_changes = max(int(level_changes), 0)

        scenario_name = g["scenario_name"].iloc[0]

        summary_rows.append({
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "ladder": ladder_name,
            "avg_mos": g["selected_mos"].mean(),
            "min_mos": g["selected_mos"].min(),
            "avg_selected_bitrate_mbps": g["selected_bitrate_mbps"].mean(),
            "total_data_mb": g["data_mb"].sum(),
            "switch_count": level_changes,
            "share_time_below_mos_3": (g["selected_mos"] < 3.0).mean(),
            "intervals": len(g),
            "mos_per_mbps": g["selected_mos"].mean() / g["selected_bitrate_mbps"].mean()
        })

    summary = pd.DataFrame(summary_rows)

    # Расчёт разниц относительно YouTube внутри каждого сценария.
    for scenario_id in summary["scenario_id"].unique():
        mask = summary["scenario_id"] == scenario_id
        y_row = summary[mask & (summary["ladder"] == "youtube")].iloc[0]

        for idx, row in summary[mask].iterrows():
            summary.loc[idx, "data_saving_vs_youtube_percent"] = (
                (y_row["total_data_mb"] - row["total_data_mb"]) /
                y_row["total_data_mb"] * 100.0
            )
            summary.loc[idx, "bitrate_saving_vs_youtube_percent"] = (
                (y_row["avg_selected_bitrate_mbps"] - row["avg_selected_bitrate_mbps"]) /
                y_row["avg_selected_bitrate_mbps"] * 100.0
            )
            summary.loc[idx, "avg_mos_delta_vs_youtube"] = row["avg_mos"] - y_row["avg_mos"]
            summary.loc[idx, "efficiency_delta_vs_youtube"] = row["mos_per_mbps"] - y_row["mos_per_mbps"]

    summary.to_csv(OUTPUT_DIR / "scenarios_ladder_selection_summary.csv", index=False, encoding="utf-8-sig")

    comparison = make_comparison_table(summary)
    comparison.to_csv(OUTPUT_DIR / "scenarios_comparison_matrix.csv", index=False, encoding="utf-8-sig")

    print(f"\nСохранено: {OUTPUT_DIR / 'scenarios_ladder_selection_timeline.csv'}")
    print(f"Сохранено: {OUTPUT_DIR / 'scenarios_ladder_selection_summary.csv'}")
    print(f"Сохранено: {OUTPUT_DIR / 'scenarios_comparison_matrix.csv'}")

    print("\nСводная таблица сценариев:")
    print(comparison.to_string(index=False))

    return timeline, summary, comparison


def make_comparison_table(summary):
    rows = []

    for scenario_id in summary["scenario_id"].unique():
        g = summary[summary["scenario_id"] == scenario_id]

        p = g[g["ladder"] == "proposed"].iloc[0]
        y = g[g["ladder"] == "youtube"].iloc[0]

        rows.append({
            "scenario_id": scenario_id,
            "scenario_name": p["scenario_name"],
            "proposed_avg_mos": p["avg_mos"],
            "youtube_avg_mos": y["avg_mos"],
            "mos_delta_proposed_minus_youtube": p["avg_mos"] - y["avg_mos"],
            "proposed_avg_bitrate_mbps": p["avg_selected_bitrate_mbps"],
            "youtube_avg_bitrate_mbps": y["avg_selected_bitrate_mbps"],
            "bitrate_saving_proposed_percent": (
                (y["avg_selected_bitrate_mbps"] - p["avg_selected_bitrate_mbps"]) /
                y["avg_selected_bitrate_mbps"] * 100.0
            ),
            "proposed_total_data_mb": p["total_data_mb"],
            "youtube_total_data_mb": y["total_data_mb"],
            "data_saving_proposed_percent": (
                (y["total_data_mb"] - p["total_data_mb"]) /
                y["total_data_mb"] * 100.0
            ),
            "proposed_switch_count": p["switch_count"],
            "youtube_switch_count": y["switch_count"],
            "proposed_mos_per_mbps": p["mos_per_mbps"],
            "youtube_mos_per_mbps": y["mos_per_mbps"],
            "efficiency_gain_proposed_percent": (
                (p["mos_per_mbps"] - y["mos_per_mbps"]) /
                y["mos_per_mbps"] * 100.0
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. Графики
# ============================================================
#
# Логика визуализации приведена к тексту раздела 4.6:
# - основные рисунки 4.9 и 4.10 строятся по сценарию S2;
# - во всех подписях используется термин "перцептивная лестница";
# - сводный рисунок 4.11 строится как две связанные панели,
#   а не как две оси на одном поле. Так высота столбцов не вводит
#   в заблуждение: экономия данных и ΔMOS имеют разные физические шкалы.
#
# Основные файлы для вставки в диплом:
#   figure_4_9_S2_channel_and_bitrate.png
#   figure_4_10_S2_quality_timeline.png
#   figure_4_11_scenarios_data_saving_and_mos_delta.png
#
# Дополнительно сохраняются графики по всем сценариям:
#   S0_high_4k_step_channel_and_bitrate.png
#   S0_high_4k_step_quality_timeline.png
#   ...
# ============================================================

MAIN_SCENARIO_ID = "S2_upper_mid_1440_step"

PLOT_DIR_NAME = "figures_scenarios"

COLOR_YOUTUBE = "#1f77b4"
COLOR_PERCEPTUAL = "#ff7f0e"
COLOR_CHANNEL = "#333333"
COLOR_SAVING = "#2e7d32"
COLOR_DELTA = "#d95f02"
COLOR_GRID = "#b0b0b0"

SCENARIO_SHORT_LABELS = {
    "S0_high_4k_step": "S0",
    "S1_baseline_1080_step": "S1",
    "S2_upper_mid_1440_step": "S2",
    "S3_medium_720_step": "S3",
    "S4_low_480_step": "S4",
    "S5_high_4k_gradual": "S5",
    "S6_upper_mid_1440_gradual": "S6",
    "S7_upper_mid_1440_oscillating": "S7",
}

SCENARIO_AXIS_LABELS = {
    "S0_high_4k_step": "S0\n4K\nрезко",
    "S1_baseline_1080_step": "S1\n1080p\nрезко",
    "S2_upper_mid_1440_step": "S2\n1440p/P2\nрезко",
    "S3_medium_720_step": "S3\n720p/P3\nрезко",
    "S4_low_480_step": "S4\n480p/P4\nрезко",
    "S5_high_4k_gradual": "S5\n4K\nплавно",
    "S6_upper_mid_1440_gradual": "S6\n1440p/P2\nплавно",
    "S7_upper_mid_1440_oscillating": "S7\n1440p/P2\nколебания",
}

SCENARIO_READABLE_NAMES = {
    "S0_high_4k_step": "S0: область 4K, резкое падение канала",
    "S1_baseline_1080_step": "S1: область 1080p/P2, резкое падение канала",
    "S2_upper_mid_1440_step": "S2: область 1440p/P2, резкое падение канала",
    "S3_medium_720_step": "S3: область 720p/P3, резкое падение канала",
    "S4_low_480_step": "S4: область 480p/P4, резкое падение канала",
    "S5_high_4k_gradual": "S5: область 4K, плавное изменение канала",
    "S6_upper_mid_1440_gradual": "S6: область 1440p/P2, плавное изменение канала",
    "S7_upper_mid_1440_oscillating": "S7: область 1440p/P2, повторные колебания",
}


def get_plot_dir():
    out = OUTPUT_DIR / PLOT_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_filename(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["_", "-", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def style_axes(ax, major_grid=True):
    ax.set_facecolor("white")
    if major_grid:
        ax.grid(True, which="major", color=COLOR_GRID, alpha=0.28, linewidth=0.8)
        ax.grid(True, which="minor", color=COLOR_GRID, alpha=0.12, linewidth=0.5)
        ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_phase_background(ax, scenario_profile, y_top=None):
    """
    Показывает фазы канала полупрозрачными зонами.
    Это помогает читать график в презентации: где высокий канал,
    где падение и где восстановление.
    """
    phase_colors = {
        "стабильный канал": "#e8f1fb",
        "падение канала": "#fff0e6",
        "восстановление": "#eaf7ea",
        "плавное падение": "#fff0e6",
        "низкий канал": "#fbe9e7",
        "плавное восстановление": "#eaf7ea",
        "первое падение": "#fff0e6",
        "первое восстановление": "#eaf7ea",
        "второе падение": "#fff0e6",
        "второе восстановление": "#eaf7ea",
    }

    blocks = []
    cur_phase = scenario_profile.iloc[0]["phase_ru"]
    start_t = scenario_profile.iloc[0]["time_start_sec"]

    for i in range(1, len(scenario_profile)):
        next_phase = scenario_profile.iloc[i]["phase_ru"]
        if next_phase != cur_phase:
            end_t = scenario_profile.iloc[i - 1]["time_end_sec"]
            blocks.append((cur_phase, start_t, end_t))
            cur_phase = next_phase
            start_t = scenario_profile.iloc[i]["time_start_sec"]

    blocks.append((cur_phase, start_t, scenario_profile.iloc[-1]["time_end_sec"]))

    for phase, x0, x1 in blocks:
        ax.axvspan(x0, x1, color=phase_colors.get(phase, "#f5f5f5"), alpha=0.55, zorder=0)
        if y_top is not None and (x1 - x0) >= 10:
            ax.text(
                (x0 + x1) / 2,
                y_top,
                phase,
                ha="center",
                va="top",
                fontsize=9,
                color="#444444"
            )


def step_arrays(df, value_col):
    x = np.r_[df["time_start_sec"].values, df["time_end_sec"].iloc[-1]]
    y = np.r_[df[value_col].values, df[value_col].iloc[-1]]
    return x, y


def switch_rows(df, value_col="selected_level_id"):
    """
    Возвращает строки, где выбранный уровень изменился.
    Первая строка также включается.
    """
    rows = []
    prev = None

    for _, row in df.iterrows():
        value = row[value_col]
        if value != prev:
            rows.append(row)
            prev = value

    return rows


def bitrate_label(row):
    return (
        f"{row['selected_label']}\n"
        f"{row['selected_bitrate_mbps']:.2f} Мбит/с"
    )


def mos_label(row):
    return (
        f"{row['selected_label']}\n"
        f"MOS={row['selected_mos']:.2f}"
    )


def plot_scenario_channel_and_bitrate(timeline, scenario_id, main_figure=False):
    """
    График профиля канала и выбранных битрейтов.

    Для основного текста диплома main_figure=True используется для S2
    и сохраняет файл figure_4_9_S2_channel_and_bitrate.png.
    """
    plot_dir = get_plot_dir()

    g_s = timeline[timeline["scenario_id"] == scenario_id].copy()
    if g_s.empty:
        raise ValueError(f"Нет данных для сценария {scenario_id}")

    scenario_title = SCENARIO_READABLE_NAMES.get(scenario_id, g_s["scenario_name"].iloc[0])

    profile = (
        g_s[g_s["ladder"] == "proposed"]
        [["time_start_sec", "time_end_sec", "bandwidth_mbps", "phase_ru"]]
        .drop_duplicates()
        .sort_values("time_start_sec")
    )

    youtube = g_s[g_s["ladder"] == "youtube"].sort_values("time_start_sec")
    perceptual = g_s[g_s["ladder"] == "proposed"].sort_values("time_start_sec")

    fig, ax = plt.subplots(figsize=(13.8, 6.6), facecolor="white")

    ymax = max(
        float(profile["bandwidth_mbps"].max()),
        float(youtube["selected_bitrate_mbps"].max()),
        float(perceptual["selected_bitrate_mbps"].max())
    ) * 1.22

    add_phase_background(ax, profile, y_top=ymax * 0.98)

    x_prof, y_prof = step_arrays(profile, "bandwidth_mbps")
    ax.step(
        x_prof,
        y_prof,
        where="post",
        linewidth=3.2,
        color=COLOR_CHANNEL,
        label="Доступная пропускная способность канала",
        zorder=4
    )

    for df, label, color in [
        (youtube, "YouTube: выбранный битрейт", COLOR_YOUTUBE),
        (perceptual, "Перцептивная лестница: выбранный битрейт", COLOR_PERCEPTUAL),
    ]:
        x, y = step_arrays(df, "selected_bitrate_mbps")
        ax.step(
            x,
            y,
            where="post",
            linewidth=3.0,
            color=color,
            label=label,
            zorder=5
        )

        # Подписываем только точки переключения, чтобы не перегружать график.
        for row in switch_rows(df, "selected_level_id"):
            marker_x = row["time_start_sec"]
            marker_y = row["selected_bitrate_mbps"]
            ax.scatter([marker_x], [marker_y], s=46, color=color, edgecolor="white", linewidth=0.8, zorder=7)

            # Разносим подписи YouTube и перцептивной лестницы.
            # Правило специально сделано простым и стабильным:
            # YouTube чаще подписывается выше линии, перцептивная лестница — ниже.
            # Если подпись уехала бы за нижнюю/верхнюю границу, переносим её внутрь.
            if color == COLOR_YOUTUBE:
                y_offset = ymax * 0.060
                va = "bottom"
                if marker_y + y_offset > ymax * 0.92:
                    y_offset = -ymax * 0.075
                    va = "top"
            else:
                y_offset = -ymax * 0.075
                va = "top"
                if marker_y + y_offset < ymax * 0.075:
                    y_offset = ymax * 0.060
                    va = "bottom"

            # Для основного рисунка 4.9 (S2) низкий интервал 30-60 с
            # подписываем по центру области: YouTube сверху, перцептивную лестницу снизу.
            # Так плашки 720p/1280x720 и 980x551 не налезают друг на друга.
            if (
                scenario_id == MAIN_SCENARIO_ID
                and abs(float(row["time_start_sec"]) - 30.0) < 1e-9
                and abs(float(row["time_end_sec"]) - 60.0) < 1e-9
            ):
                x_text = (float(row["time_start_sec"]) + float(row["time_end_sec"])) / 2
                ha = "center"
                if color == COLOR_YOUTUBE:
                    y_offset = ymax * 0.070
                    va = "bottom"
                else:
                    y_offset = -ymax * 0.085
                    va = "top"
            else:
                # В начале графика подпись ставим правее точки, в конце — левее,
                # чтобы она не обрезалась краем изображения.
                if marker_x > SIMULATION_DURATION_SEC * 0.78:
                    x_text = marker_x - 1.2
                    ha = "right"
                else:
                    x_text = marker_x + 0.9
                    ha = "left"

            ax.text(
                x_text,
                marker_y + y_offset,
                bitrate_label(row),
                ha=ha,
                va=va,
                fontsize=8.5,
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.96),
                zorder=8,
                clip_on=True
            )

    # Подписываем верхний и нижний уровни канала.
    high = float(profile["bandwidth_mbps"].max())
    low = float(profile["bandwidth_mbps"].min())

    high_y = min(high + ymax * 0.028, ymax * 0.93)
    ax.text(
        1,
        high_y,
        f"верхний уровень канала: {high:.2f} Мбит/с",
        fontsize=9,
        color=COLOR_CHANNEL,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_CHANNEL, alpha=0.96),
        clip_on=True
    )

    low_row = profile.loc[profile["bandwidth_mbps"].idxmin()]
    low_x = float(low_row["time_start_sec"]) + 1.0
    low_y = max(low + ymax * 0.055, ymax * 0.12)
    low_y = min(low_y, ymax * 0.78)
    ax.text(
        low_x,
        low_y,
        f"нижний уровень канала: {low:.2f} Мбит/с",
        fontsize=9,
        color=COLOR_CHANNEL,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_CHANNEL, alpha=0.96),
        clip_on=True
    )

    ax.set_xlim(0, SIMULATION_DURATION_SEC)
    ax.set_ylim(0, ymax * 1.04)
    ax.set_xlabel("Время, с", fontsize=11)
    ax.set_ylabel("Пропускная способность / выбранный битрейт, Мбит/с", fontsize=11)
    ax.set_title(
        f"Профиль канала и выбранные битрейты\n{scenario_title}",
        fontsize=15,
        fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.96)
    style_axes(ax)

    fig.tight_layout()

    scenario_file = plot_dir / f"{safe_filename(scenario_id)}_channel_and_bitrate.png"
    plt.savefig(scenario_file, dpi=250, bbox_inches="tight")

    if main_figure:
        main_file = plot_dir / "figure_4_9_S2_channel_and_bitrate.png"
        plt.savefig(main_file, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_scenario_quality_timeline(timeline, scenario_id, main_figure=False):
    """
    График изменения расчётной MOS во времени.

    Для основного текста диплома main_figure=True используется для S2
    и сохраняет файл figure_4_10_S2_quality_timeline.png.
    """
    plot_dir = get_plot_dir()

    g_s = timeline[timeline["scenario_id"] == scenario_id].copy()
    if g_s.empty:
        raise ValueError(f"Нет данных для сценария {scenario_id}")

    scenario_title = SCENARIO_READABLE_NAMES.get(scenario_id, g_s["scenario_name"].iloc[0])

    profile = (
        g_s[g_s["ladder"] == "proposed"]
        [["time_start_sec", "time_end_sec", "bandwidth_mbps", "phase_ru"]]
        .drop_duplicates()
        .sort_values("time_start_sec")
    )

    youtube = g_s[g_s["ladder"] == "youtube"].sort_values("time_start_sec")
    perceptual = g_s[g_s["ladder"] == "proposed"].sort_values("time_start_sec")

    fig, ax = plt.subplots(figsize=(13.8, 6.4), facecolor="white")

    add_phase_background(ax, profile, y_top=5.13)

    for df, label, color in [
        (youtube, "YouTube", COLOR_YOUTUBE),
        (perceptual, "Перцептивная лестница", COLOR_PERCEPTUAL),
    ]:
        x, y = step_arrays(df, "selected_mos")
        ax.step(
            x,
            y,
            where="post",
            linewidth=3.0,
            color=color,
            label=label,
            zorder=5
        )

        # Точки ставим в центре временного интервала, чтобы они не казались
        # смещёнными относительно ступени.
        centers = (df["time_start_sec"].values + df["time_end_sec"].values) / 2
        ax.scatter(
            centers,
            df["selected_mos"].values,
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=7
        )

        # Подписываем только уровни при переключениях.
        for row in switch_rows(df, "selected_level_id"):
            x_pos = (row["time_start_sec"] + row["time_end_sec"]) / 2
            y_pos = row["selected_mos"]

            # Разносим подписи YouTube и перцептивной лестницы и держим
            # их внутри рабочей области графика.
            y_offset = 0.10 if color == COLOR_YOUTUBE else -0.13
            va = "bottom" if y_offset > 0 else "top"

            # Для основного рисунка 4.10 (S2) низкий интервал 30-60 с
            # подписываем по центру области: YouTube сверху, перцептивную лестницу снизу.
            if (
                scenario_id == MAIN_SCENARIO_ID
                and abs(float(row["time_start_sec"]) - 30.0) < 1e-9
                and abs(float(row["time_end_sec"]) - 60.0) < 1e-9
            ):
                x_text = (float(row["time_start_sec"]) + float(row["time_end_sec"])) / 2
                ha = "center"
                if color == COLOR_YOUTUBE:
                    y_offset = 0.10
                    va = "bottom"
                else:
                    y_offset = -0.13
                    va = "top"
            else:
                x_text = x_pos + 0.8
                ha = "left"

            ax.text(
                x_text,
                y_pos + y_offset,
                mos_label(row),
                ha=ha,
                va=va,
                fontsize=8.5,
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.96),
                zorder=8,
                clip_on=False
            )

    # Динамическая шкала MOS. Раньше была жёсткая шкала 3.8-4.8,
    # из-за чего сценарии S0/S3/S4 ломали изображение: подписи уходили
    # за пределы графика и появлялись огромные белые поля.
    mos_values = np.r_[youtube["selected_mos"].values, perceptual["selected_mos"].values]
    mos_min = float(np.min(mos_values))
    mos_max = float(np.max(mos_values))
    y_low = max(1.8, np.floor((mos_min - 0.18) * 10) / 10)
    y_high = min(5.15, np.ceil((mos_max + 0.18) * 10) / 10)

    # Минимальная высота шкалы, чтобы график не выглядел слишком плоским.
    if y_high - y_low < 0.7:
        center = (y_low + y_high) / 2
        y_low = max(1.8, center - 0.35)
        y_high = min(5.15, center + 0.35)

    ax.set_xlim(0, SIMULATION_DURATION_SEC)
    ax.set_ylim(y_low, y_high)
    ax.set_yticks(np.round(np.arange(np.ceil(y_low * 10) / 10, y_high + 0.001, 0.1), 2))
    ax.set_xlabel("Время, с", fontsize=11)
    ax.set_ylabel("MOS выбранного уровня\n(выше — лучше)", fontsize=11)
    ax.set_title(
        f"Изменение воспринимаемого качества выбранных уровней\n{scenario_title}",
        fontsize=15,
        fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.96)
    style_axes(ax)

    fig.tight_layout()

    scenario_file = plot_dir / f"{safe_filename(scenario_id)}_quality_timeline.png"
    plt.savefig(scenario_file, dpi=250, bbox_inches="tight")

    if main_figure:
        main_file = plot_dir / "figure_4_10_S2_quality_timeline.png"
        plt.savefig(main_file, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_comparison_summary(comparison):
    """
    Рисунок 4.11.

    Важно: здесь НЕ используется двойная ось с двумя столбцами на одном поле,
    потому что экономия данных (%) и ΔMOS имеют разные масштабы. На одной
    двойной оси высота столбцов визуально воспринимается неверно.

    Поэтому график строится как две согласованные панели:
    - сверху: экономия данных;
    - снизу: ΔMOS.
    """
    plot_dir = get_plot_dir()

    comparison = comparison.copy()
    comparison["short_label"] = comparison["scenario_id"].map(SCENARIO_AXIS_LABELS).fillna(comparison["scenario_id"])

    x = np.arange(len(comparison))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(13.8, 8.2),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.08}
    )

    # ------------------------------------------------------------
    # Верхняя панель: экономия данных
    # ------------------------------------------------------------
    bars_saving = ax_top.bar(
        x,
        comparison["data_saving_proposed_percent"],
        width=0.62,
        color=COLOR_SAVING,
        label="Экономия данных перцептивной лестницы, %"
    )

    ax_top.axhline(0, color="#222222", linewidth=1.0)
    ax_top.set_ylabel("Экономия данных, %", fontsize=11)
    ax_top.set_title(
        "Сценарное сравнение: экономия данных и изменение MOS",
        fontsize=15,
        fontweight="bold"
    )
    ax_top.set_ylim(-16, 47)
    style_axes(ax_top)

    for rect, value in zip(bars_saving, comparison["data_saving_proposed_percent"]):
        y_pos = value + (1.2 if value >= 0 else -1.8)
        ax_top.text(
            rect.get_x() + rect.get_width() / 2,
            y_pos,
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
            color=COLOR_SAVING,
            fontweight="bold"
        )

    ax_top.text(
        0.01,
        0.95,
        "Экономия данных > 0 означает,\nчто перцептивная лестница требует меньше данных.",
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb", alpha=0.96)
    )

    # ------------------------------------------------------------
    # Нижняя панель: изменение MOS
    # ------------------------------------------------------------
    bars_delta = ax_bottom.bar(
        x,
        comparison["mos_delta_proposed_minus_youtube"],
        width=0.62,
        color=COLOR_DELTA,
        label="ΔMOS = MOS(перцептивная лестница) − MOS(YouTube)"
    )

    ax_bottom.axhline(0, color="#222222", linewidth=1.0)
    ax_bottom.set_ylabel("ΔMOS", fontsize=11)
    ax_bottom.set_xlabel("Сценарии расчётной проверки", fontsize=11)
    ax_bottom.set_ylim(-0.34, 0.13)
    style_axes(ax_bottom)

    for rect, value in zip(bars_delta, comparison["mos_delta_proposed_minus_youtube"]):
        y_pos = value + (0.012 if value >= 0 else -0.018)
        ax_bottom.text(
            rect.get_x() + rect.get_width() / 2,
            y_pos,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
            color=COLOR_DELTA,
            fontweight="bold"
        )

    ax_bottom.text(
        0.01,
        0.06,
        "ΔMOS > 0 — качество выше у перцептивной лестницы;\nΔMOS < 0 — качество выше у YouTube.",
        transform=ax_bottom.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb", alpha=0.96)
    )

    # Выделяем сценарий S2, потому что он используется на рисунках 4.9-4.10.
    if MAIN_SCENARIO_ID in comparison["scenario_id"].values:
        idx = comparison.index[comparison["scenario_id"] == MAIN_SCENARIO_ID][0]
        for ax in [ax_top, ax_bottom]:
            ax.axvspan(idx - 0.45, idx + 0.45, color="#fff4b8", alpha=0.45, zorder=0)


    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(comparison["short_label"], fontsize=9)

    # Общая легенда.
    handles_top, labels_top = ax_top.get_legend_handles_labels()
    handles_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
    ax_top.legend(
        handles_top + handles_bottom,
        labels_top + labels_bottom,
        loc="upper right",
        fontsize=9,
        framealpha=0.96
    )

    out = plot_dir / "figure_4_11_scenarios_data_saving_and_mos_delta.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Дополнительный график эффективности оставляем для приложения/презентации.
    plot_efficiency_summary(comparison)


def plot_efficiency_summary(comparison):
    plot_dir = get_plot_dir()

    comparison = comparison.copy()
    comparison["short_label"] = comparison["scenario_id"].map(SCENARIO_AXIS_LABELS).fillna(comparison["scenario_id"])

    x = np.arange(len(comparison))

    fig, ax = plt.subplots(figsize=(13.8, 5.8), facecolor="white")

    bars = ax.bar(
        x,
        comparison["efficiency_gain_proposed_percent"],
        width=0.62,
        color="#6a3d9a",
        label="Прирост эффективности MOS/Мбит/с, %"
    )

    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["short_label"], fontsize=9)
    ax.set_ylabel("Прирост эффективности, %", fontsize=11)
    ax.set_xlabel("Сценарии расчётной проверки", fontsize=11)
    ax.set_title("Сравнение эффективности использования битрейта", fontsize=15, fontweight="bold")
    ax.set_ylim(-15, 62)
    style_axes(ax)

    for rect, value in zip(bars, comparison["efficiency_gain_proposed_percent"]):
        y_pos = value + (1.3 if value >= 0 else -1.8)
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y_pos,
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
            color="#4a1d78",
            fontweight="bold"
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.96)

    out = plot_dir / "figure_4_12_scenarios_efficiency.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(timeline, comparison):
    """
    Генерирует единый комплект графиков.

    Основной комплект для диплома:
    - figure_4_9_S2_channel_and_bitrate.png
    - figure_4_10_S2_quality_timeline.png
    - figure_4_11_scenarios_data_saving_and_mos_delta.png

    Также сохраняет графики 4.9/4.10-подобного типа для всех сценариев.
    """
    for scenario_id in timeline["scenario_id"].unique():
        is_main = scenario_id == MAIN_SCENARIO_ID
        plot_scenario_channel_and_bitrate(timeline, scenario_id, main_figure=is_main)
        plot_scenario_quality_timeline(timeline, scenario_id, main_figure=is_main)

    plot_comparison_summary(comparison)

    print(f"\nГрафики сохранены в: {get_plot_dir()}")
    print("Основные файлы для раздела 4.6:")
    print(f"  {get_plot_dir() / 'figure_4_9_S2_channel_and_bitrate.png'}")
    print(f"  {get_plot_dir() / 'figure_4_10_S2_quality_timeline.png'}")
    print(f"  {get_plot_dir() / 'figure_4_11_scenarios_data_saving_and_mos_delta.png'}")


# ============================================================
# 9. Опциональные демонстрационные видео
# ============================================================

def create_demo_videos_if_needed(timeline, reference_files):
    if not CREATE_DEMO_VIDEOS:
        return

    print("Demo-видео в сценарной версии сейчас отключено по умолчанию.")
    print("Для раздела 4.6 достаточно таблиц и графиков.")


# ============================================================
# 10. MAIN
# ============================================================

def save_run_parameters():
    scenario_rows = []
    for s in SCENARIOS:
        scenario_rows.append({
            "scenario_id": s["scenario_id"],
            "scenario_name": s["scenario_name"],
            "description": s["description"],
            "high_calibration_pairs": str(s["high_calibration_pairs"]),
            "high_margin": s["high_margin"],
            "low_to_high_ratio": s["low_to_high_ratio"],
            "profile_shape": s["profile_shape"],
        })

    pd.DataFrame(scenario_rows).to_csv(
        OUTPUT_DIR / "scenarios_description.csv",
        index=False,
        encoding="utf-8-sig"
    )

    run_parameters = pd.DataFrame([{
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "reference_files": ", ".join(REFERENCE_FILES),
        "codec": CODEC,
        "crf": CRF,
        "preset": PRESET,
        "encode_sample_start_sec": ENCODE_SAMPLE_START_SEC,
        "encode_sample_duration_sec": ENCODE_SAMPLE_DURATION_SEC,
        "interval_duration_sec": INTERVAL_DURATION_SEC,
        "simulation_duration_sec": SIMULATION_DURATION_SEC,
        "safety_factor": SAFETY_FACTOR,
        "default_low_to_high_ratio": DEFAULT_LOW_TO_HIGH_RATIO,
        "load_existing_bitrates_if_available": LOAD_EXISTING_BITRATES_IF_AVAILABLE,
    }])

    run_parameters.to_csv(
        OUTPUT_DIR / "run_parameters_scenarios.csv",
        index=False,
        encoding="utf-8-sig"
    )


def main():
    print("=" * 78)
    print("СЦЕНАРНАЯ РАСЧЁТНАЯ ПРОВЕРКА ВЫБОРА УРОВНЕЙ")
    print("=" * 78)
    print(f"Среда запуска:  {'Windows' if IS_WINDOWS else ('Google Colab / Linux' if IN_COLAB else 'Linux / macOS')}")
    print(f"Входная папка:  {INPUT_DIR}")
    print(f"Выходная папка: {OUTPUT_DIR}")
    print("Ожидаемый файл: reference_1.mp4")
    print(f"Длительность кодируемого фрагмента: {ENCODE_SAMPLE_DURATION_SEC} c")
    print(f"Кодирование: CRF={CRF}, preset={PRESET}")
    print(f"Загрузка готовых битрейтов при наличии CSV: {LOAD_EXISTING_BITRATES_IF_AVAILABLE}")

    add_mos_to_ladders()
    ensure_dirs()
    check_tools()

    reference_files = find_reference_files()
    min_duration = min(get_video_info(p)["duration"] for p in reference_files)
    print(f"\nМинимальная длительность reference-видео: {min_duration:.1f} c")

    ladder_summary = get_or_create_ladder_bitrates(reference_files)
    profiles = create_all_profiles(ladder_summary, min_duration)
    timeline, summary, comparison = simulate_level_selection(ladder_summary, profiles)

    make_all_plots(timeline, comparison)
    create_demo_videos_if_needed(timeline, reference_files)
    save_run_parameters()

    print("\n" + "=" * 78)
    print("ГОТОВО")
    print("=" * 78)
    print(f"Результаты сохранены в: {OUTPUT_DIR}")
    print("\nГлавные файлы:")
    print(f"  {OUTPUT_DIR / 'channel_ladder_bitrates.csv'}")
    print(f"  {OUTPUT_DIR / 'scenarios_description.csv'}")
    print(f"  {OUTPUT_DIR / 'scenarios_channel_profiles.csv'}")
    print(f"  {OUTPUT_DIR / 'scenarios_ladder_selection_timeline.csv'}")
    print(f"  {OUTPUT_DIR / 'scenarios_ladder_selection_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'scenarios_comparison_matrix.csv'}")
    print(f"  {OUTPUT_DIR / 'figures_scenarios'}")


if __name__ == "__main__":
    main()
