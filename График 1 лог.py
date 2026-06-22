# -*- coding: utf-8 -*-
"""
График 1 ЛОГ — QoE-кривые для тестовых режимов 4K / FHD / HD.

Обновлённая логика координаты:
- исходные данные qoe_ultimate.csv содержат разрешения деградированного
  исходного 4K-видео в шкале 4096×2160;
- на защите верхняя граница интерпретируется как UHD 3840×2160;
- поэтому координата Y нормируется одинаково для всех тестовых режимов:
      width_norm = width_raw × 3840 / 4096
      height_norm = height_raw
- 4K / FHD / HD — это тестовые режимы вывода/просмотра, а не разные шкалы Y;
- линия 4096×2160 удалена;
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
OUT_PNG = "qoe_modes_lagrange_uhd_log.png"
OUT_STATS = "qoe_modes_lagrange_uhd_log_stats.csv"
OUT_INTERP = "qoe_modes_lagrange_uhd_log_interpolated_points.csv"

MOS_SCORES = [5, 4, 3, 2]
MODES = ["4K", "1080", "720"]

MODE_LABELS = {
    "4K": "тестовый режим 4K",
    "1080": "тестовый режим FHD / 1080p",
    "720": "тестовый режим HD / 720p",
}

COLORS = {
    "4K": "#1A6FBF",
    "1080": "#1E8A3A",
    "720": "#CC2020",
}

LINE_STYLES = {
    "4K": "-",
    "1080": "--",
    "720": "-.",
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

# ── 4. Расчёт статистики и ДИ ─────────────────────────────────
# Среднее точки считается по очищенным наблюдениям.
# ДИ считается консервативно: сначала среднее внутри участника,
# затем 95% ДИ по межсубъектному разбросу.
S = {}
stats_rows = []

for mode in MODES:
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
    x_pos = 0.78

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
                fontsize=7.0 if res == 240 else 7.5,
                color="#777777",
                style="italic",
                zorder=3,
            )

# ── 8. Функция построения одного режима ───────────────────────
def draw_single(ax, mode, show_ylabel=True):
    color = COLORS[mode]
    ls_style = LINE_STYLES[mode]

    xv = np.array(MOS_SCORES, dtype=float)
    yv = np.array([S[mode][m]["mean"] for m in MOS_SCORES])
    cv = np.array([S[mode][m]["ci"] for m in MOS_SCORES])
    labels = [S[mode][m]["label"] for m in MOS_SCORES]

    X_R, X_L = 5.55, 0.72

    ax.set_xlim(X_R, X_L)
    ax.set_ylim(LOG_Y_BOT, LOG_Y_TOP)
    ax.set_yscale("log")

    apply_grid(ax)
    draw_reference_levels(ax, LOG_Y_BOT, LOG_Y_TOP)

    # Реальные наблюдения.
    mask_mode = df[MODE_COL].astype(str) == mode
    scatter_x, scatter_y = [], []

    for mos in MOS_SCORES:
        tmp = df[mask_mode & (df["MOS"] == mos)]
        for val in tmp["width"].dropna():
            if val >= LOG_Y_BOT:
                scatter_x.append(mos + np.random.uniform(-0.05, 0.05))
                scatter_y.append(val)

    ax.scatter(scatter_x, scatter_y, s=14, color="#808080", alpha=0.18, edgecolors="none", zorder=2)

    # Проекции к осям.
    for xi, yi in zip(xv, yv):
        if np.isnan(yi):
            continue
        ax.hlines(yi, xi, X_R, colors=color, linestyles=":", lw=1.0, alpha=0.5, zorder=2)
        ax.vlines(xi, LOG_Y_BOT, yi, colors=color, linestyles=":", lw=1.0, alpha=0.5, zorder=2)

    # Интерполяционный полином Лагранжа.
    xd, yd = lagrange_curve(xv, yv, num=500)
    valid_plot = yd > 0
    ax.plot(
        xd[valid_plot],
        yd[valid_plot],
        color=color,
        ls=ls_style,
        lw=2.8,
        solid_capstyle="round",
        alpha=0.95,
        zorder=4,
    )

    # Средние точки.
    ax.plot(
        xv,
        yv,
        marker="o",
        color=color,
        markersize=11,
        lw=0,
        markeredgecolor="white",
        markeredgewidth=2.2,
        zorder=5,
    )

    # Доверительные интервалы.
    ax.errorbar(
        xv,
        yv,
        yerr=cv,
        fmt="none",
        ecolor=color,
        elinewidth=2.5,
        capsize=8,
        capthick=2.5,
        zorder=6,
    )

    # Подписи средних значений.
    for xi, yi, lbl in zip(xv, yv, labels):
        if np.isnan(yi):
            continue

        if xi == 5:
            offset_text = (0, -26)
            va_text = "top"
        else:
            offset_text = (0, 12)
            va_text = "bottom"

        ax.annotate(
            lbl,
            xy=(xi, yi),
            xytext=offset_text,
            textcoords="offset points",
            ha="center",
            va=va_text,
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor=color, lw=1.2, alpha=0.95),
            zorder=7,
        )

    ax.set_xticks(MOS_SCORES)
    ax.set_xticklabels([MOS_LABELS[m].replace("\n", " ") for m in MOS_SCORES], fontsize=10)

    apply_log_y_ticks(ax, LOG_Y_BOT, LOG_Y_TOP)

    ax.set_title(MODE_LABELS[mode], fontsize=13, fontweight="bold", color=color, pad=12)
    ax.set_xlabel("Субъективная оценка MOS (ITU-R BT.500)", fontsize=10, labelpad=8)

    if show_ylabel:
        ax.set_ylabel("Нормированное пороговое разрешение, ширина (пикс., лог. шкала)", fontsize=10, labelpad=8)

# ── 9. Функция построения сводного графика ────────────────────
def draw_combined(ax):
    X_R, X_L = 5.55, 0.72

    ax.set_xlim(X_R, X_L)
    ax.set_ylim(LOG_Y_BOT, LOG_Y_TOP)
    ax.set_yscale("log")

    apply_grid(ax)
    draw_reference_levels(ax, LOG_Y_BOT, LOG_Y_TOP)

    LANE_HALF_WIDTH = 0.13

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

    # Сдвиги используются только для визуального разнесения кривых.
    OFFSETS = {
        "720": 0.11,
        "1080": 0.0,
        "4K": -0.11,
    }

    # Тонкие горизонтальные проекции.
    for mode in MODES:
        color = COLORS[mode]
        for mos in MOS_SCORES:
            xi = float(mos) + OFFSETS[mode]
            yi = S[mode][mos]["mean"]
            if np.isnan(yi):
                continue
            ax.hlines(yi, xi, X_R, colors=color, linestyles=":", lw=1.0, alpha=0.4, zorder=2)

    # Отрисовка интерполяционных полиномов Лагранжа.
    for mode in MODES:
        color = COLORS[mode]
        ls_style = LINE_STYLES[mode]

        x_model = np.array(MOS_SCORES, dtype=float)
        yv = np.array([S[mode][m]["mean"] for m in MOS_SCORES])
        cv = np.array([S[mode][m]["ci"] for m in MOS_SCORES])

        offset = OFFSETS[mode]
        xv = x_model + offset
        xd_model, yd = lagrange_curve(x_model, yv, num=400)
        xd = xd_model + offset

        valid_plot = yd > 0
        ax.plot(
            xd[valid_plot],
            yd[valid_plot],
            color=color,
            ls=ls_style,
            lw=2.5,
            alpha=0.95,
            solid_capstyle="round",
            zorder=3,
        )

        ax.plot(
            xv,
            yv,
            marker="o",
            color=color,
            markersize=9,
            lw=0,
            markeredgecolor="white",
            markeredgewidth=2.0,
            zorder=5,
            label=MODE_LABELS[mode],
        )

        ax.errorbar(xv, yv, yerr=cv, fmt="none", ecolor=color, elinewidth=2.2, capsize=6, capthick=2.2, zorder=6)

        for xi, yi, mos in zip(xv, yv, MOS_SCORES):
            if np.isnan(yi):
                continue

            if mode == "720":
                dx, dy = -13, 14
                ha, va = "right", "bottom"
            elif mode == "1080":
                dx, dy = 0, -18
                ha, va = "center", "top"
            else:
                dx, dy = 13, 14
                ha, va = "left", "bottom"

            ax.annotate(
                f"{yi:.0f}",
                xy=(xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=8,
                color=color,
                fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor=color, lw=0.8, alpha=0.95),
                zorder=7,
            )

    ax.set_xticks(MOS_SCORES)
    ax.set_xticklabels([MOS_LABELS[m] for m in MOS_SCORES], fontsize=10, fontweight="bold")

    minor_ticks = []
    for m in MOS_SCORES:
        minor_ticks.extend([m - LANE_HALF_WIDTH, m + LANE_HALF_WIDTH])

    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    ax.tick_params(axis="x", which="minor", colors="#A0A0A0", length=6, width=1.2)
    ax.tick_params(axis="x", which="major", length=0, pad=8)

    apply_log_y_ticks(ax, LOG_Y_BOT, LOG_Y_TOP)

    ax.set_title("Сводный график — тестовые режимы", fontsize=13, fontweight="bold", color="#222222", pad=12)
    ax.set_xlabel("Выделенные зоны субъективных оценок (MOS)", fontsize=10, labelpad=4)
    ax.set_ylabel("Нормированное пороговое разрешение, ширина (пикс., лог. шкала)", fontsize=10, labelpad=8)

    ax.legend(
        title="Тестовый режим",
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.95,
        edgecolor="#BBBBBB",
        borderaxespad=0.0,
    )

# ── 10. Диагностика ───────────────────────────────────────────
n_p = df[SUBJECT_COL].nunique()
n_s = df["Сессия"].nunique() if "Сессия" in df.columns else "—"

print("\nКонтроль верхней границы:")
for mode in MODES:
    mode_max = df.loc[df[MODE_COL].astype(str) == mode, "width"].max()
    mean_max = max(S[mode][m]["mean"] for m in MOS_SCORES if not np.isnan(S[mode][m]["mean"]))
    print(f"  {MODE_LABELS[mode]}: max наблюдения = {mode_max:.1f} px, max среднего = {mean_max:.1f} px")

global_max = df["width"].max()
if global_max > DISPLAY_MAX_WIDTH + 1e-9:
    raise ValueError(f"Ошибка нормировки: найдено значение выше {DISPLAY_MAX_WIDTH}px: {global_max}")
else:
    print(f"  ✓ Все значения ≤ {DISPLAY_MAX_WIDTH}px")

print("\nДиагностика интерполяции Лагранжа:")
interp_rows = []
for mode in MODES:
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

# ── 11. Компоновка панели 2×2 ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(22, 16), facecolor="white")
fig.patch.set_facecolor("white")

fig.suptitle(
    "QoE-лестница: субъективная оценка качества видео vs нормированное пороговое разрешение (логарифмическая шкала)\n"
    f"Участников: {n_p}  |  Сессий: {n_s}  |  "
    "Координата: деградированное 4K-видео, приведённое к UHD-шкале 3840×2160  |  "
    "Интерполяция: полином 3-й степени по 4 опорным точкам  |  "
    "Вертикальные интервалы: 95% ДИ среднего значения по участникам |  "
    "Штрихпунктир: эталонные уровни разрешения",
    fontsize=12,
    y=1.01,
    color="#222222",
)

draw_single(axes[0, 0], "4K", show_ylabel=True)
draw_single(axes[0, 1], "1080", show_ylabel=True)
draw_single(axes[1, 0], "720", show_ylabel=True)
draw_combined(axes[1, 1])

# Оставляем место справа для вынесенной легенды сводного графика.
plt.tight_layout(h_pad=5, w_pad=4, rect=[0, 0, 0.94, 1])

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="none")

print(f"\n✓ График сохранён: {OUT_PNG}")
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
