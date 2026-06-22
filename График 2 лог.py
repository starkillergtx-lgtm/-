# -*- coding: utf-8 -*-
"""
График 2 ЛОГ — формирование базовой QoE-лестницы по новой компромиссной логике.

Обновлённая логика координаты:
- исходные данные qoe_ultimate.csv содержат разрешения деградированного
  исходного 4K-видео в шкале 4096×2160;
- верхняя граница для защиты приводится к UHD 3840×2160;
- поэтому координата Y нормируется одинаково для всех тестовых режимов:
      width_norm = width_raw × 3840 / 4096
      height_norm = height_raw
- 4K / FHD / HD — это тестовые режимы вывода/просмотра, а не разные шкалы Y;
- линия 4096×2160 удалена;
- Main-кривая строится как среднее трёх режимных кривых уже в нормированной
  UHD-шкале 3840×2160;
- логарифмическая шкала меняет только отображение оси Y, а не расчёт средних,
  доверительных интервалов или интерполяции.

Интерполяция:
- глобальный полином 3-й степени по четырём опорным MOS-точкам;
- явная форма Лагранжа;
- PCHIP / сплайн / регрессия не используются.
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── 0. Настройки ──────────────────────────────────────────────
np.random.seed(42)

SOURCE_WIDTH = 4096
DISPLAY_MAX_WIDTH = 3840
DISPLAY_MAX_HEIGHT = 2160

INPUT_CSV = "qoe_ultimate.csv"
OUT_PNG = "qoe_average_ladder_lagrange_uhd_log.png"
OUT_STATS = "qoe_average_ladder_lagrange_uhd_log_stats.csv"
OUT_INTERP = "qoe_average_ladder_lagrange_uhd_log_interpolated_points.csv"

MOS_SCORES = [5, 4, 3, 2]
RAW_MODES = ["4K", "1080", "720"]
PLOT_MODES = ["4K", "1080", "720", "Main"]

MODE_LABELS = {
    "4K": "тестовый режим 4K",
    "1080": "тестовый режим FHD / 1080p",
    "720": "тестовый режим HD / 720p",
    "Main": "основная QoE-кривая",
}

COLORS = {
    "4K": "#1A6FBF",
    "1080": "#1E8A3A",
    "720": "#CC2020",
    "Main": "#111111",
}

LINE_STYLES = {
    "4K": "-",
    "1080": "--",
    "720": "-.",
    "Main": "-",
}

MOS_LABELS = {
    5: "5\n(отлично)",
    4: "4\n(хорошо)",
    3: "3\n(удовл.)",
    2: "2\n(плохо)",
}

# 4096×2160 специально исключён: верхняя граница шкалы — UHD 3840×2160.
REFERENCE_LEVELS = [
    (3840, "3840×2160  (UHD 4K — верхняя граница шкалы)"),
    (2560, "2560×1440  (QHD / 2K)"),
    (1920, "1920×1080  (FHD)"),
    (1280, "1280×720   (HD 720p)"),
    (854,  " 854×480   (SD 480p)"),
    (640,  " 640×360   (360p)"),
    (426,  " 426×240   (240p)"),
    (240,  "240 px     (нижняя граница лог. шкалы)"),
]

LOG_Y_BOT = 150
LOG_Y_TOP = 4500
LOG_TICKS = [240, 426, 640, 854, 1280, 1920, 2560, 3840]

print("Метод интерполяции: глобальный полином 3-й степени по 4 опорным точкам")
print("Форма вычисления: явная интерполяция Лагранжа")
print("PCHIP / сплайн / регрессия: НЕ используется")
print(f"Нормировка разрешений: 4096×2160 → {DISPLAY_MAX_WIDTH}×{DISPLAY_MAX_HEIGHT}")
print("Режимы 4K/FHD/HD используются как условия просмотра, а не как разные шкалы Y")
print("Main-кривая считается как среднее трёх тестовых режимов в одной UHD-шкале")
print("Логарифмическая шкала применяется только к отображению оси Y")

# ── 1. Загрузка CSV ───────────────────────────────────────────
def load_csv():
    """
    1) Локально / VS Code: положите qoe_ultimate.csv рядом со скриптом
       или измените INPUT_CSV.
    2) В Google Colab: если файла рядом нет, появится upload-окно.
    """
    candidate_paths = [
        INPUT_CSV,
        "qoe_ultimate(3).csv",
        "/mnt/data/qoe_ultimate.csv",
        "/mnt/data/qoe_ultimate(3).csv",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            print(f"\nЗагружаю файл: {path}")
            return pd.read_csv(path, encoding="utf-8-sig")

    try:
        from google.colab import files
        print("\nФайл qoe_ultimate.csv не найден рядом со скриптом.")
        print("Загрузите qoe_ultimate.csv:")
        uploaded = files.upload()
        frames = []
        for fname, content in uploaded.items():
            tmp = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            frames.append(tmp)
            print(f"  ✓ {fname}: {len(tmp)} строк")
        return pd.concat(frames, ignore_index=True)
    except Exception as e:
        raise FileNotFoundError(
            "Не найден qoe_ultimate.csv. Положите CSV рядом со скриптом "
            "или запустите код в Colab и загрузите файл через upload."
        ) from e


df_all = load_csv()

# ── 2. Фильтрация по статусу ──────────────────────────────────
if "Статус" in df_all.columns:
    df = df_all[df_all["Статус"] == "OK"].copy()
    excl = len(df_all) - len(df)
    print(f"Фильтр по Статус='OK': {len(df)} строк включено, {excl} исключено")
else:
    def tc_sec_f(tc):
        try:
            if str(tc) == "None":
                return np.nan
            p = str(tc).split(":")
            return int(p[0]) * 60 + int(p[1])
        except Exception:
            return np.nan

    df_all["_tc"] = df_all["Таймкод"].apply(tc_sec_f)
    df = df_all[~((df_all["MOS"] == 5) & (df_all["_tc"] > 30))].copy()
    print(f"Стандартный фильтр: {len(df)} строк")

# ── 3. Предобработка разрешений ───────────────────────────────
def get_dim(res, idx):
    try:
        res_clean = str(res).replace("x", "×")
        return int(res_clean.split("×")[idx])
    except Exception:
        return np.nan


MODE_COL = "Formatted_screen_size" if "Formatted_screen_size" in df.columns else "Формат_экрана"
SUBJECT_COL = "Испытуемый"

if MODE_COL not in df.columns:
    raise KeyError("Не найден столбец с тестовым режимом: Formatted_screen_size или Формат_экрана")
if SUBJECT_COL not in df.columns:
    raise KeyError("Не найден столбец Испытуемый")

# Исходная координата из CSV.
df["raw_width"] = df["Разрешение"].apply(lambda r: get_dim(r, 0))
df["raw_height"] = df["Разрешение"].apply(lambda r: get_dim(r, 1))

# Компромиссная нормировка: все режимы остаются в одной UHD-шкале.
df["width"] = df["raw_width"] * DISPLAY_MAX_WIDTH / SOURCE_WIDTH
df["height"] = df["raw_height"]

# Защита от артефактов / округлений.
df["width"] = df["width"].clip(upper=DISPLAY_MAX_WIDTH)
df["height"] = df["height"].clip(upper=DISPLAY_MAX_HEIGHT)

# Строка нормированного разрешения для таблиц/подписей.
df["Разрешение_норм"] = (
    df["width"].round().astype("Int64").astype(str)
    + "×"
    + df["height"].round().astype("Int64").astype(str)
)

# ── 4. Расчёт статистики, ДИ и Main-кривой ───────────────────
# Среднее точки считается по очищенным наблюдениям.
# ДИ считается консервативно: сначала среднее внутри участника,
# затем 95% ДИ по межсубъектному разбросу.
S = {}
stats_rows = []

for mode in RAW_MODES:
    S[mode] = {}

    for mos in MOS_SCORES:
        mask = (df[MODE_COL].astype(str) == mode) & (df["MOS"] == mos)
        group = df.loc[mask, [SUBJECT_COL, "width", "height"]].dropna()

        if len(group) == 0:
            mw, mh, ci_w, ci_h, n_obs, n_subj = np.nan, np.nan, 0.0, 0.0, 0, 0
        else:
            mw = group["width"].mean()
            mh = group["height"].mean()
            n_obs = len(group)

            subj_w = group.groupby(SUBJECT_COL)["width"].mean().dropna()
            subj_h = group.groupby(SUBJECT_COL)["height"].mean().dropna()
            n_subj = len(subj_w)

            if n_subj > 1:
                ci_w = stats.t.ppf(0.975, df=n_subj - 1) * stats.sem(subj_w)
                ci_h = stats.t.ppf(0.975, df=n_subj - 1) * stats.sem(subj_h)
            else:
                ci_w, ci_h = 0.0, 0.0

        label = f"{int(round(mw))}×{int(round(mh))}" if not np.isnan(mw) else "—"

        S[mode][mos] = {
            "mean": mw,
            "height": mh,
            "ci": ci_w,
            "ci_height": ci_h,
            "n": n_obs,
            "n_subj": n_subj,
            "label": label,
        }

        stats_rows.append({
            "row_type": "mode",
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "MOS": mos,
            "mean_width_px_norm": mw,
            "mean_height_px_norm": mh,
            "ci95_width_px": ci_w,
            "ci95_height_px": ci_h,
            "n_observations": n_obs,
            "n_subjects": n_subj,
            "label": label,
        })

# Main = среднее трёх режимных кривых в одной и той же нормированной UHD-шкале.
S["Main"] = {}
for mos in MOS_SCORES:
    widths = [S[mode][mos]["mean"] for mode in RAW_MODES if not np.isnan(S[mode][mos]["mean"])]
    heights = [S[mode][mos]["height"] for mode in RAW_MODES if not np.isnan(S[mode][mos]["height"])]

    if len(widths) > 0:
        mw = float(np.mean(widths))
        mh = float(np.mean(heights)) if len(heights) > 0 else np.nan
        label = f"{int(round(mw))}×{int(round(mh))}" if not np.isnan(mh) else f"{int(round(mw))}"
    else:
        mw, mh, label = np.nan, np.nan, "—"

    S["Main"][mos] = {
        "mean": mw,
        "height": mh,
        "ci": 0.0,
        "ci_height": 0.0,
        "n": 0,
        "n_subj": 0,
        "label": label,
    }

    stats_rows.append({
        "row_type": "main_average",
        "mode": "Main",
        "mode_label": MODE_LABELS["Main"],
        "MOS": mos,
        "mean_width_px_norm": mw,
        "mean_height_px_norm": mh,
        "ci95_width_px": 0.0,
        "ci95_height_px": 0.0,
        "n_observations": 0,
        "n_subjects": 0,
        "label": label,
    })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")

# ── 5. Интерполяционный полином Лагранжа ──────────────────────
def lagrange_eval(x_points, y_points, x_eval):
    """
    Явная интерполяция Лагранжа.
    Для четырёх опорных точек получается единственный полином 3-й степени.
    """
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    valid = ~(np.isnan(x_points) | np.isnan(y_points))
    x_points = x_points[valid]
    y_points = y_points[valid]

    order = np.argsort(x_points)
    x_points = x_points[order]
    y_points = y_points[order]

    y_eval = np.zeros_like(x_eval, dtype=float)

    for i in range(len(x_points)):
        basis = np.ones_like(x_eval, dtype=float)
        for j in range(len(x_points)):
            if i != j:
                basis *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        y_eval += y_points[i] * basis

    return y_eval


def lagrange_curve(x_points, y_points, num=500):
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)

    valid = ~(np.isnan(x_points) | np.isnan(y_points))
    x_valid = x_points[valid]

    xd = np.linspace(np.min(x_valid), np.max(x_valid), num)
    yd = lagrange_eval(x_points, y_points, xd)
    yd = np.clip(yd, LOG_Y_BOT, DISPLAY_MAX_WIDTH)
    return xd, yd


def lagrange_value_at(x_points, y_points, x_query):
    return float(np.clip(lagrange_eval(x_points, y_points, np.array([x_query]))[0], LOG_Y_BOT, DISPLAY_MAX_WIDTH))

# ── 6. Оформление координатной сетки ──────────────────────────
def apply_grid(ax):
    ax.set_facecolor("#FDFDFD")
    ax.grid(True, which="major", color="#E2E2E2", lw=0.9, ls="-", zorder=0)
    ax.grid(True, which="minor", color="#EEEEEE", lw=0.5, ls="-", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(1.2)
        ax.spines[sp].set_color("#444444")


def apply_log_y_ticks(ax, y_bot, y_top):
    visible_ticks = [t for t in LOG_TICKS if y_bot <= t <= y_top]

    ax.yaxis.set_major_locator(mticker.FixedLocator(visible_ticks))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if y_bot <= v <= y_top else "")
    )
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


# ── 7. Эталонные линии ────────────────────────────────────────
def draw_reference_levels(ax, y_bot, y_top):
    x_pos = 0.92

    for res, label in REFERENCE_LEVELS:
        if y_bot < res < y_top:
            if res == 3840:
                line_alpha = 0.55
                text_y = res * 0.965
                text_va = "top"
            elif res == 240:
                line_alpha = 0.55
                text_y = res * 0.93
                text_va = "top"
            else:
                line_alpha = 0.75
                text_y = res * 1.015
                text_va = "bottom"

            ax.axhline(res, color="#BBBBBB", lw=1.0, ls="-.", zorder=1, alpha=line_alpha)
            ax.text(
                x_pos,
                text_y,
                label,
                ha="right",
                va=text_va,
                fontsize=7.2 if res == 240 else 8.0,
                color="#777777",
                style="italic",
                zorder=3,
            )

# ── 8. Подготовка графика ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")

X_R, X_L = 5.55, 0.88

ax.set_xlim(X_R, X_L)
ax.set_ylim(LOG_Y_BOT, LOG_Y_TOP)
ax.set_yscale("log")

apply_grid(ax)
draw_reference_levels(ax, LOG_Y_BOT, LOG_Y_TOP)

LANE_HALF_WIDTH = 0.24

for mos in MOS_SCORES:
    ax.axvspan(
        mos - LANE_HALF_WIDTH,
        mos + LANE_HALF_WIDTH,
        facecolor="#F0F0F0",
        edgecolor="#D5D5D5",
        lw=1.0,
        alpha=0.75,
        zorder=1,
    )
    ax.axvline(mos, color="#999999", lw=1.2, ls=":", zorder=1)

# Три режимные кривые расположены как на сводном графике:
# 4K = -0.11, 1080 = 0.00, 720 = +0.11.
# Чёрная Main-кривая добавлена четвёртым рядом справа.
OFFSETS = {
    "4K": -0.11,
    "1080": 0.00,
    "720": 0.11,
    "Main": 0.22,
}

# Тонкие горизонтальные проекции для режимных средних.
for mode in RAW_MODES:
    color = COLORS[mode]
    for mos in MOS_SCORES:
        xi = float(mos) + OFFSETS[mode]
        yi = S[mode][mos]["mean"]
        if np.isnan(yi):
            continue
        ax.hlines(yi, xi, X_R, colors=color, linestyles=":", lw=1.0, alpha=0.4, zorder=2)

# Ручное позиционирование проблемных подписей при необходимости.
CUSTOM_LABEL_POSITIONS = {}

# Отрисовка кривых.
# Полином строится по настоящим MOS-точкам [5, 4, 3, 2].
# Горизонтальные сдвиги используются только для визуального разнесения рядов.
for mode in PLOT_MODES:
    is_main = (mode == "Main")
    color = COLORS[mode]
    ls_style = LINE_STYLES[mode]

    x_model = np.array(MOS_SCORES, dtype=float)
    yv = np.array([S[mode][m]["mean"] for m in MOS_SCORES])
    cv = np.array([S[mode][m]["ci"] for m in MOS_SCORES])

    offset = OFFSETS[mode]
    xv_display = x_model + offset

    xd_model, yd = lagrange_curve(x_model, yv, num=500)
    xd_display = xd_model + offset

    line_w = 3.5 if is_main else 2.0
    line_alpha = 1.0 if is_main else 0.75
    z_line = 9 if is_main else 3

    label_name = "Основная перцептивная лестница (среднее тестовых режимов)" if is_main else MODE_LABELS[mode]

    valid_plot = yd > 0
    ax.plot(
        xd_display[valid_plot],
        yd[valid_plot],
        color=color,
        ls=ls_style,
        lw=line_w,
        alpha=line_alpha,
        solid_capstyle="round",
        zorder=z_line,
        label=label_name,
    )

    m_face = color if is_main else "white"
    m_edge = "white" if is_main else color
    m_size = 10 if is_main else 8
    z_mark = 10 if is_main else 5

    ax.plot(
        xv_display,
        yv,
        marker="o",
        color=color,
        markersize=m_size,
        lw=0,
        markerfacecolor=m_face,
        markeredgecolor=m_edge,
        markeredgewidth=2.2,
        zorder=z_mark,
    )

    if not is_main:
        ax.errorbar(
            xv_display,
            yv,
            yerr=cv,
            fmt="none",
            ecolor=color,
            alpha=0.75,
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            zorder=4,
        )

    for xi, yi, mos in zip(xv_display, yv, MOS_SCORES):
        if np.isnan(yi):
            continue

        if (mode, mos) in CUSTOM_LABEL_POSITIONS:
            dx, dy = CUSTOM_LABEL_POSITIONS[(mode, mos)]
            ha, va = "center", "center"
        else:
            if mode == "Main":
                dx, dy = 10, 12
                ha, va = "left", "bottom"
            elif mode == "720":
                dx, dy = -8, -14
                ha, va = "right", "top"
            elif mode == "1080":
                dx, dy = 8, -14
                ha, va = "left", "top"
            elif mode == "4K":
                dx, dy = 12, 12
                ha, va = "left", "bottom"

        font_w = "black" if is_main else "bold"
        f_size = 9 if is_main else 8

        ax.annotate(
            f"{yi:.0f}",
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=f_size,
            color=color,
            fontweight=font_w,
            bbox=dict(
                boxstyle="square,pad=0.2",
                facecolor="white",
                edgecolor=color,
                lw=1.2 if is_main else 0.8,
                alpha=0.95,
            ),
            zorder=11,
        )

ax.set_xticks(MOS_SCORES)
ax.set_xticklabels([MOS_LABELS[m] for m in MOS_SCORES], fontsize=11, fontweight="bold")

minor_ticks = []
for m in MOS_SCORES:
    minor_ticks.extend([m - LANE_HALF_WIDTH, m + LANE_HALF_WIDTH])

ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
ax.tick_params(axis="x", which="minor", colors="#A0A0A0", length=8, width=1.2)
ax.tick_params(axis="x", which="major", length=0, pad=10)

apply_log_y_ticks(ax, LOG_Y_BOT, LOG_Y_TOP)

n_p = df[SUBJECT_COL].nunique()
n_s = df["Сессия"].nunique() if "Сессия" in df.columns else "—"

# Диагностика: значения полинома по всем рядам.
print("\nКонтроль верхней границы:")
global_max = df["width"].max()
print(f"  max наблюдения после нормировки = {global_max:.1f} px")
if global_max > DISPLAY_MAX_WIDTH + 1e-9:
    raise ValueError(f"Ошибка нормировки: найдено значение выше {DISPLAY_MAX_WIDTH}px: {global_max}")
else:
    print(f"  ✓ Все значения ≤ {DISPLAY_MAX_WIDTH}px")

print("\nДиагностика интерполяции Лагранжа:")
interp_rows = []
for mode in PLOT_MODES:
    x_check = np.array(MOS_SCORES, dtype=float)
    y_check = np.array([S[mode][m]["mean"] for m in MOS_SCORES], dtype=float)

    print(f"\n{MODE_LABELS[mode]}:")
    for mos_q in [4.5, 3.5, 2.5]:
        w_q = lagrange_value_at(x_check, y_check, mos_q)
        print(f"  MOS={mos_q:.1f}: ширина ≈ {w_q:.1f} px, округление: {int(round(w_q))} px")
        interp_rows.append({
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "MOS_interpolated": mos_q,
            "width_px_norm": w_q,
            "width_px_rounded": int(round(w_q)),
        })

pd.DataFrame(interp_rows).to_csv(OUT_INTERP, index=False, encoding="utf-8-sig")

ax.set_title(
    "Формирование базовой QoE-лестницы: усреднение тестовых режимов (логарифмическая шкала)\n",
    fontsize=15,
    fontweight="bold",
    color="#222222",
    pad=25,
)

plt.figtext(
    0.5,
    0.90,
    f"Участников: {n_p}  |  Сессий: {n_s}  |  "
    "Координата: деградированное 4K-видео, приведённое к UHD-шкале 3840×2160  |  "
    "Интерполяция: полином Лагранжа 3-й степени",
    ha="center",
    fontsize=10.5,
    color="#555555",
)

ax.set_xlabel("Выделенные зоны субъективных оценок (MOS)", fontsize=12, labelpad=8)
ax.set_ylabel("Нормированное пороговое разрешение, ширина (пикс., лог. шкала)", fontsize=12, labelpad=12)

leg = ax.legend(
    fontsize=10.5,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    framealpha=0.95,
    edgecolor="#BBBBBB",
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=1.5,
)

for line in leg.get_lines():
    line.set_linewidth(3.0)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")

print(f"\n✓ График формирования лестницы сохранён: {OUT_PNG}")
print(f"✓ Таблица средних и ДИ сохранена: {OUT_STATS}")
print(f"✓ Интерполированные точки сохранены: {OUT_INTERP}")

try:
    from google.colab import files
    files.download(OUT_PNG)
    files.download(OUT_STATS)
    files.download(OUT_INTERP)
except Exception:
    pass

plt.show()
