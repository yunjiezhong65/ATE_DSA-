# -*- coding: utf-8 -*-
"""
DSA ATE Board Consistency Analyzer v3.5
板内模型拟合、板内判定、全局模型、全局判定四层解耦
新增：全板卡汇总明细表导出、多挡位多板卡统计与异常诊断矩阵导出
"""
import sys, re, os, json, math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from scipy.stats import chi2, gaussian_kde
from sklearn.covariance import MinCovDet

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse

APP_VERSION = '3.5'

# 正则匹配规则
FILE_NAME_PAT = re.compile(
    r'^No(?P<no>\d+)\s+(?P<bp>BP[-\w]+)\s+(?P<jc>JC\d+)\s+(?P<idx>\d+)(?:st|nd|rd|th)\s+(?P<res>pass|fail)\.log$',
    re.I
)
FVMV_PAT = re.compile(
    r"DPS FVMV Calibrated\. OC\d+\s+(?P<range>\S+)\s+(?P<impedance>\S+)\s+chn:\s*(?P<chn>\d+)\s+"
    r"LFo:\s*(?P<LFo>[-+]?(?:\d+(?:\.\d*)?|\.\d+)m?V)\s+LADC:\s*(?P<LADC>[-+]?(?:\d+(?:\.\d*)?|\.\d+)m?V)\s+"
    r"LMeter:\s*(?P<LMeter>[-+]?(?:\d+(?:\.\d*)?|\.\d+)m?V)\s+HFo:\s*(?P<HFo>[-+]?(?:\d+(?:\.\d*)?|\.\d+)m?V)\s+"
    r"HADC:\s*(?P<HADC>[-+]?(?:\d+(?:\.\d*)?|\.\d+)m?V)\s+HMeter:\s*(?P<HMeter>[-+]?(?:\d+(?:\.\d*)?|\.\d+)m?V)",
    re.I
)
MI_PAT = re.compile(
    r"DPS MI Calibrated\. OC\d+\s+(?P<range>\S+)\s+(?P<impedance>\S+)\s+chn:\s*(?P<chn>\d+)\s+"
    r"LFV:\s*(?P<LFV>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:mV|V))\s+LADC:\s*(?P<LADC>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:nA|uA|mA|A))\s+"
    r"LMeter:\s*(?P<LMeter>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:nA|uA|mA|A))\s+HFV:\s*(?P<HFV>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:mV|V))\s+"
    r"HADC:\s*(?P<HADC>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:nA|uA|mA|A))\s+HMeter:\s*(?P<HMeter>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:nA|uA|mA|A))",
    re.I
)

SPEC_K_LIMIT = (0.99, 1.01)
SPEC_B_LIMIT = (-0.1, 0.1)
BOARD_SURVIVAL_PASS = 0.90
BOARD_SURVIVAL_WARN = 0.75
BOARD_CENTER_WARN_SIGMA = 3.0
BOARD_SPREAD_WARN_RATIO = 1.5
COV_EPS = 1e-12

def voltage_to_v(s):
    m = re.match(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+))(mV|V)$', str(s).strip(), re.I)
    if not m: return np.nan
    v = float(m.group(1))
    return v * 1e-3 if m.group(2).lower() == 'mv' else v

def current_to_ua(s):
    m = re.match(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+))(nA|uA|mA|A)$', str(s).strip(), re.I)
    if not m: return np.nan
    v = float(m.group(1)); u = m.group(2).lower()
    return {'na': v / 1000.0, 'ua': v, 'ma': v * 1000.0, 'a': v * 1e6}[u]

def get_current_unit_scale(range_str):
    """自适应电流单位解析：支持 500mA、2P5MA、1P2A 等"""
    r = str(range_str).upper().strip()
    if 'NA' in r:
        return 'nA', 1e3
    elif 'UA' in r:
        return 'uA', 1.0
    elif 'MA' in r:
        return 'mA', 1e-3
    elif re.search(r'(?:\d|P)A\b', r) or r.endswith('A'):
        return 'mA', 1e-3
    return 'mA', 1e-3

def parse_range_nominal(range_str, default=1.0):
    """解析量程标称绝对值（用于计算偏移容限）"""
    if not range_str or pd.isna(range_str):
        return default
    r = str(range_str).upper().replace(' ', '')
    m = re.search(r'([0-9]+(?:P[0-9]+|\.[0-9]+)?)([A-Z]+)', r)
    if m:
        val_str, unit = m.groups()
        val_str = val_str.replace('P', '.')
        try:
            val = float(val_str)
            if unit == 'A':
                return val * 1000.0  # 转为 mA 对应
            return val
        except Exception:
            return default
    try:
        return float(r)
    except Exception:
        return default

def ensure_cov(cov):
    try: cov = np.asarray(cov, float)
    except Exception: return np.eye(2) * COV_EPS
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)): return np.eye(2) * COV_EPS
    cov = (cov + cov.T) / 2
    try:
        e, v = np.linalg.eigh(cov)
        e = np.maximum(e, COV_EPS)
        return v @ np.diag(e) @ v.T
    except Exception:
        return np.eye(2) * COV_EPS

def md2(X, center, cov):
    X = np.asarray(X, float)
    d = X - np.asarray(center, float)
    inv = np.linalg.pinv(ensure_cov(cov))
    out = np.einsum('ij,jk,ik->i', d, inv, d)
    out[~np.isfinite(out)] = np.inf
    return out

def cutoff(conf):
    return float(chi2.ppf(min(max(float(conf), 0.500001), 0.999999), 2))

def ellipse_params(center, cov, c):
    try:
        e, v = np.linalg.eigh(ensure_cov(cov))
        ix = np.argsort(e)[::-1]; e = e[ix]; v = v[:, ix]
        s = math.sqrt(max(c, COV_EPS))
        return (float(center[0]), float(center[1]), 2 * s * math.sqrt(max(e[0], COV_EPS)),
                2 * s * math.sqrt(max(e[1], COV_EPS)), float(math.degrees(math.atan2(v[1, 0], v[0, 0]))))
    except Exception:
        return None

def parse_single_log(path):
    vs = []; cs = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for line in lines:
        m = FVMV_PAT.search(line)
        if m:
            d = m.groupdict()
            d['chn'] = int(d['chn'])
            la, lm, ha, hm = [voltage_to_v(d[x]) for x in ('LADC', 'LMeter', 'HADC', 'HMeter')]
            da = ha - la
            d['k'] = (hm - lm) / da if abs(da) > COV_EPS else np.nan
            d['b'] = lm - d['k'] * la if np.isfinite(d['k']) else np.nan
            d['b_unit'] = 'V'
            d['measurement_type'] = '电压'
            vs.append(d)
            continue
        m = MI_PAT.search(line)
        if m:
            d = m.groupdict()
            d['chn'] = int(d['chn'])
            la, lm, ha, hm = [current_to_ua(d[x]) for x in ('LADC', 'LMeter', 'HADC', 'HMeter')]
            if not all(np.isfinite(z) for z in (la, lm, ha, hm)):
                continue
            d.update(LADC_uA=la, LMeter_uA=lm, HADC_uA=ha, HMeter_uA=hm)
            da = ha - la
            k_val = (hm - lm) / da if abs(da) > COV_EPS else np.nan
            b_val_ua = lm - k_val * la if np.isfinite(k_val) else np.nan
            unit, scale = get_current_unit_scale(d.get('range', ''))
            d['k'] = k_val
            d['b'] = b_val_ua * scale if np.isfinite(b_val_ua) else np.nan
            d['b_unit'] = unit
            d['measurement_type'] = '电流'
            cs.append(d)
    # 不去重，如实保留所有读取出的行
    return pd.DataFrame(vs), pd.DataFrame(cs)

def load_logs(paths):
    out = {}
    for path in paths:
        fn = os.path.basename(path)
        stem = os.path.splitext(fn)[0]
        m = FILE_NAME_PAT.match(fn)
        if m:
            g = m.groupdict()
            no = int(g['no']); bp = g['bp']; jc = g['jc']; idx = int(g['idx']); res = g['res'].lower()
            board_id = f'{bp}{jc}'
        else:
            res = 'fail' if 'fail' in fn.lower() else 'pass'
            no = 0; idx = 1; board_id = stem
        try:
            dv, dc = parse_single_log(path)
        except Exception:
            continue
        if dv.empty and dc.empty:
            continue
        ds = {}
        if not dv.empty:
            ds['电压校准'] = dv
        if not dc.empty:
            for r in sorted(dc['range'].dropna().unique(), key=str):
                ds[f'电流-{r}'] = dc[dc['range'] == r].reset_index(drop=True)
        rid = stem
        if rid in out:
            rid = f"{stem}_{len(out)}"
        out[rid] = {
            'test_run_id': rid,
            'board_id': board_id,
            'test_idx': idx,
            'test_result': res,
            'no_num': no,
            'file_name': fn,
            'display_name': stem,
            'file_path': os.path.abspath(path),
            'datasets': ds
        }
    return out

# ----------------- 诊断与报表生成 -----------------
def diagnose_channels_text(df, range_str='', is_voltage=False):
    """根据理想值 (k=1, b=0) 及 ±0.5% 偏差判定通道状态"""
    if df.empty or 'k' not in df.columns or 'b' not in df.columns:
        return "无数据"
    unit = 'V' if is_voltage else str(df['b_unit'].iloc[0] if 'b_unit' in df.columns else '')
    nominal = 1.0 if is_voltage else parse_range_nominal(range_str, default=1.0)
    tol_b = abs(nominal) * 0.005  # ±0.5% 容限
    issues = []
    
    for _, row in df.iterrows():
        chn = int(row['chn']) if ('chn' in row and pd.notna(row['chn'])) else '?'
        k = row.get('k', np.nan)
        b = row.get('b', np.nan)
        errs = []
        if np.isfinite(k):
            if k > 1.005: errs.append(f"k偏大:{k:.4f}")
            elif k < 0.995: errs.append(f"k偏小:{k:.4f}")
        else:
            errs.append("k缺失")
        if np.isfinite(b):
            if b > tol_b: errs.append(f"b偏大:{b:+.3f}{unit}")
            elif b < -tol_b: errs.append(f"b偏小:{b:+.3f}{unit}")
        else:
            errs.append("b缺失")
        if errs:
            issues.append(f"Ch{chn}({';'.join(errs)})")
            
    if not issues:
        return "正常"
    if len(issues) > 10:
        return f"{' | '.join(issues[:10])} 等共{len(issues)}通道异常"
    return " | ".join(issues)

def export_consolidated_details_csv(tests, save_path):
    """生成表一：所有板卡明细合并 CSV（含电压与电流两大区段）"""
    all_vs = []
    all_cs = []
    for rid, t in tests.items():
        if '电压校准' in t['datasets']:
            df_v = t['datasets']['电压校准'].copy()
            df_v.insert(0, '测试运行ID', rid)
            df_v.insert(1, '日志文件名', t['file_name'])
            df_v.insert(2, '板卡标识', t['board_id'])
            all_vs.append(df_v)
        cs_list = [v for k, v in t['datasets'].items() if k.startswith('电流-')]
        if cs_list:
            df_c = pd.concat(cs_list, ignore_index=True)
            df_c.insert(0, '测试运行ID', rid)
            df_c.insert(1, '日志文件名', t['file_name'])
            df_c.insert(2, '板卡标识', t['board_id'])
            all_cs.append(df_c)
            
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write("# ==============================================================================\n")
        f.write("#                           全部板卡：电压校准明细数据汇总\n")
        f.write("# ==============================================================================\n")
        if all_vs:
            pd.concat(all_vs, ignore_index=True).to_csv(f, index=False)
        else:
            f.write("无电压校准数据\n")
        f.write("\n\n# ==============================================================================\n")
        f.write("#                           全部板卡：电流校准明细数据汇总\n")
        f.write("# ==============================================================================\n")
        if all_cs:
            pd.concat(all_cs, ignore_index=True).to_csv(f, index=False)
        else:
            f.write("无电流校准数据\n")

def export_matrix_statistics_csv(tests, save_path):
    """
    生成全板统计与诊断矩阵 CSV：
    - 横轴（列）：各板卡编号 / 日志名称
    - 纵轴（行）：电压及电流各挡位的 k, b 统计量（最大、最小、平均、方差）
                 并将原先的备注替换为标明公式的「归一化零偏离散度」
    """
    boards = list(tests.keys())
    board_names = [tests[b]['display_name'] for b in boards]
    
    # 提取所有出现的电流挡位名称（按字符排序）
    all_mi_ranges = sorted(list(set(
        k.replace('电流-', '') for t in tests.values() for k in t['datasets'] if k.startswith('电流-')
    )), key=str)
    
    rows = []
    
    # ------------------ 1. 电压校准区段 ------------------
    v_formula_label = '归一化零偏离散度 (%FS = σ_b / 1.0V × 100%)'
    v_metrics = [
        'k 最大值', 'k 最小值', 'k 平均值', 'k 方差',
        'b 最大值', 'b 最小值', 'b 平均值', 'b 方差',
        v_formula_label
    ]
    
    for m in v_metrics:
        row = {'项目 / 统计量': f"[电压校准] {m}"}
        for b_id in boards:
            t = tests[b_id]
            df = t['datasets'].get('电压校准', pd.DataFrame())
            if df.empty or 'k' not in df or 'b' not in df:
                row[t['display_name']] = np.nan
                continue
            kv = df['k'].dropna().values
            bv = df['b'].dropna().values
            
            if m == 'k 最大值': row[t['display_name']] = np.nanmax(kv) if len(kv) else np.nan
            elif m == 'k 最小值': row[t['display_name']] = np.nanmin(kv) if len(kv) else np.nan
            elif m == 'k 平均值': row[t['display_name']] = np.nanmean(kv) if len(kv) else np.nan
            elif m == 'k 方差': row[t['display_name']] = np.nanvar(kv, ddof=1) if len(kv) > 1 else 0.0
            elif m == 'b 最大值': row[t['display_name']] = np.nanmax(bv) if len(bv) else np.nan
            elif m == 'b 最小值': row[t['display_name']] = np.nanmin(bv) if len(bv) else np.nan
            elif m == 'b 平均值': row[t['display_name']] = np.nanmean(bv) if len(bv) else np.nan
            elif m == 'b 方差': row[t['display_name']] = np.nanvar(bv, ddof=1) if len(bv) > 1 else 0.0
            elif m == v_formula_label:
                # 电压基准归一化计算：σ_b / 1.0V * 100%
                if len(bv) > 1:
                    std_b = np.nanstd(bv, ddof=1)
                    val = (std_b / 1.0) * 100.0
                    row[t['display_name']] = f"{val:.4f}%"
                else:
                    row[t['display_name']] = "0.0000%" if len(bv) == 1 else np.nan
        rows.append(row)
        
    # 插入空行进行模块分隔
    rows.append({c: '' for c in ['项目 / 统计量'] + board_names})
    
    # ------------------ 2. 电流校准分挡位区段 ------------------
    for r in all_mi_ranges:
        ds_name = f'电流-{r}'
        nominal_val = parse_range_nominal(r, default=1.0)  # 对应 b 单位的标称量程值
        i_formula_label = f'归一化零偏离散度 (%FS = σ_b / {r} × 100%)'
        
        i_metrics = [
            'k 最大值', 'k 最小值', 'k 平均值', 'k 方差',
            'b 最大值', 'b 最小值', 'b 平均值', 'b 方差',
            i_formula_label
        ]
        
        for m in i_metrics:
            row = {'项目 / 统计量': f"[{ds_name}] {m}"}
            for b_id in boards:
                t = tests[b_id]
                df = t['datasets'].get(ds_name, pd.DataFrame())
                if df.empty or 'k' not in df or 'b' not in df:
                    row[t['display_name']] = np.nan
                    continue
                kv = df['k'].dropna().values
                bv = df['b'].dropna().values
                
                if m == 'k 最大值': row[t['display_name']] = np.nanmax(kv) if len(kv) else np.nan
                elif m == 'k 最小值': row[t['display_name']] = np.nanmin(kv) if len(kv) else np.nan
                elif m == 'k 平均值': row[t['display_name']] = np.nanmean(kv) if len(kv) else np.nan
                elif m == 'k 方差': row[t['display_name']] = np.nanvar(kv, ddof=1) if len(kv) > 1 else 0.0
                elif m == 'b 最大值': row[t['display_name']] = np.nanmax(bv) if len(bv) else np.nan
                elif m == 'b 最小值': row[t['display_name']] = np.nanmin(bv) if len(bv) else np.nan
                elif m == 'b 平均值': row[t['display_name']] = np.nanmean(bv) if len(bv) else np.nan
                elif m == 'b 方差': row[t['display_name']] = np.nanvar(bv, ddof=1) if len(bv) > 1 else 0.0
                elif m == i_formula_label:
                    # 电流归一化离散度计算：σ_b / FS * 100%
                    if len(bv) > 1:
                        std_b = np.nanstd(bv, ddof=1)
                        val = (std_b / nominal_val) * 100.0
                        row[t['display_name']] = f"{val:.4f}%"
                    else:
                        row[t['display_name']] = "0.0000%" if len(bv) == 1 else np.nan
            rows.append(row)
            
    df_matrix = pd.DataFrame(rows)
    df_matrix.to_csv(save_path, index=False, encoding='utf-8-sig')

def export_csv(tests, root='校准参数导出文件夹'):
    base = Path(os.getcwd()) / root
    base.mkdir(parents=True, exist_ok=True)
    n = 0
    for rid, t in tests.items():
        sub = f"No{t['no_num']}_{t['test_idx']}th_{t['test_result']}" if t['no_num'] > 0 else t['display_name']
        folder = base / t['board_id'] / sub
        folder.mkdir(parents=True, exist_ok=True)
        if '电压校准' in t['datasets']:
            d = t['datasets']['电压校准']
            cols = [c for c in ['range', 'impedance', 'chn', 'LFo', 'LADC', 'LMeter', 'HFo', 'HADC', 'HMeter', 'k', 'b', 'b_unit'] if c in d]
            d[cols].to_csv(folder / '电压校准.csv', index=False, encoding='utf-8-sig')
            n += 1
        cs = [v for k, v in t['datasets'].items() if k.startswith('电流-')]
        if cs:
            d = pd.concat(cs, ignore_index=True)
            cols = [c for c in ['range', 'impedance', 'chn', 'LFV', 'LADC', 'LMeter', 'HFV', 'HADC', 'HMeter', 'LADC_uA', 'LMeter_uA', 'HADC_uA', 'HMeter_uA', 'k', 'b', 'b_unit'] if c in d]
            d[cols].to_csv(folder / '电流校准.csv', index=False, encoding='utf-8-sig')
            n += 1
    messagebox.showinfo('导出成功', f'成功导出各板卡独立校准文件（共 {n} 个）到：\n{base}')

# ----------------- 统计模型核心 -----------------
def fit_rmcd(X, fit_conf=0.975, support=0.85):
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), axis=1)]
    n = len(X)
    med = np.median(X, axis=0) if n else np.array([np.nan, np.nan])
    if n == 0:
        return dict(center=med, cov=np.eye(2) * COV_EPS, raw_median=med, fit_core=np.array([], dtype=bool), fit_d2=np.array([]), method='empty')
    if n < 5:
        cov = np.cov(X, rowvar=False) if n >= 2 else np.eye(2) * COV_EPS
        center = med; d = md2(X, center, cov); core = np.ones(n, bool)
        return dict(center=center, cov=ensure_cov(cov), raw_median=med, fit_core=core, fit_d2=d, method='median_fallback')
    frac = max(5.0 / n, min(float(support), 1.0))
    try:
        m = MinCovDet(support_fraction=frac, random_state=42).fit(X)
        rc = m.location_; cv = ensure_cov(m.covariance_); method = 'MinCovDet'
    except Exception:
        rc = med; cv = ensure_cov(np.cov(X, rowvar=False)); method = 'median_cov_fallback'
    d = md2(X, rc, cv)
    finite = np.isfinite(d)
    q = float(np.quantile(d[finite], min(max(fit_conf, 0.50), 0.9999))) if finite.any() else np.inf
    core = finite & (d <= q)
    if core.sum() < 5:
        ix = np.argsort(np.where(finite, d, np.inf))
        core[:] = False; core[ix[:min(n, 5)]] = True
    center = np.median(X[core], axis=0)
    cov = ensure_cov(np.cov(X[core], rowvar=False) if core.sum() >= 2 else np.eye(2) * COV_EPS)
    d2 = md2(X, center, cov)
    return dict(center=center, cov=cov, raw_median=med, fit_core=core, fit_d2=d2, method=method)

def fit_board_models(tests, ds, fit_conf, support):
    models = {}
    for rid, t in tests.items():
        if ds not in t['datasets']: continue
        df = t['datasets'][ds].copy()
        df['test_run_id'] = rid; df['board_id'] = t['board_id']; df['file_name'] = t['file_name']
        df['display_name'] = t['display_name']; df['test_idx'] = t['test_idx']; df['test_result'] = t['test_result']
        good = np.all(np.isfinite(df[['k', 'b']].values), axis=1)
        x = df.loc[good, ['k', 'b']].values
        m = fit_rmcd(x, fit_conf, support)
        d = np.full(len(df), np.inf); d[good] = m['fit_d2']
        df['step1_d2'] = d; df['step1_model_core'] = False
        ix = np.where(good)[0]
        df.loc[df.index[ix[m['fit_core']]], 'step1_model_core'] = True
        models[rid] = {'df': df, 'center': m['center'], 'cov': m['cov'], 'raw_median': m['raw_median'], 'fit_conf': fit_conf, 'method': m['method']}
    return models

def board_filter(models, conf):
    c = cutoff(conf); fs = []
    for rid, m in models.items():
        d = m['df'].copy()
        d['step1_is_ok'] = np.isfinite(d['step1_d2']) & (d['step1_d2'] <= c)
        d['board_cutoff'] = c
        fs.append(d[d['step1_is_ok']].copy())
    return (pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()), c

def fit_global_balanced(df, fit_conf, support):
    if df.empty: return None
    gs = []
    for _, g in df.groupby('test_run_id', sort=False):
        g = g[np.all(np.isfinite(g[['k', 'b']].values), axis=1)]
        if len(g): gs.append(g)
    if not gs: return None
    target = max(5, int(np.median([len(g) for g in gs])))
    parts = []
    for g in gs: parts.append(g.sample(target, random_state=42) if len(g) > target else g)
    bal = pd.concat(parts, ignore_index=True)
    m = fit_rmcd(bal[['k', 'b']].values, fit_conf, support)
    return {'model': m, 'fit_df': bal, 'target_per_run': target, 'run_count': len(parts)}

def apply_global(df, model, conf):
    if df.empty or model is None: return pd.DataFrame(), cutoff(conf)
    d = df.copy()
    d['global_d2'] = md2(d[['k', 'b']].values, model['center'], model['cov'])
    d['global_cutoff'] = cutoff(conf)
    d['global_is_ok'] = d['global_d2'] <= d['global_cutoff']
    return d, d['global_cutoff'].iloc[0]

def board_stats(df, center, cov):
    rows = []; inv = np.linalg.pinv(ensure_cov(cov))
    for rid, g in df.groupby('test_run_id', sort=False):
        total = len(g); s1 = int(g['step1_is_ok'].sum()); good = int(g['global_is_ok'].sum())
        rate = good / total if total else 0.0; cond = good / s1 if s1 else 0.0
        xy = g.loc[g['global_is_ok'], ['k', 'b']].values
        bc = np.median(xy, axis=0) if len(xy) else np.array([np.nan, np.nan])
        cd = float((bc - center) @ inv @ (bc - center)) if np.all(np.isfinite(bc)) else np.nan
        cs = math.sqrt(max(cd, 0.0)) if np.isfinite(cd) else np.nan
        spr = math.sqrt(max(np.trace(ensure_cov(np.cov(xy, rowvar=False))) if len(xy) >= 3 else np.nan, 0.0)) if len(xy) >= 3 else np.nan
        gs = math.sqrt(max(float(np.trace(ensure_cov(cov))), COV_EPS))
        sr = spr / gs if np.isfinite(spr) and gs > 0 else np.nan
        status = 'PASS'; reason = []
        if rate < BOARD_SURVIVAL_WARN: status = 'FAIL'; reason.append('通道存活率低')
        elif rate < BOARD_SURVIVAL_PASS: status = 'WARN'; reason.append('通道存活率偏低')
        if np.isfinite(cs) and cs > BOARD_CENTER_WARN_SIGMA: status = 'FAIL'; reason.append('中心明显偏离全局')
        if np.isfinite(sr) and sr > BOARD_SPREAD_WARN_RATIO and status == 'PASS': status = 'WARN'; reason.append('板内离散度偏大')
        rows.append(dict(
            test_run_id=rid, board_id=g['board_id'].iloc[0], file_name=g['file_name'].iloc[0],
            display_name=g['display_name'].iloc[0], total_channels=total, step1_good=s1, global_good=good,
            survival_rate=rate, conditional_survival_rate=cond, board_center_k=bc[0] if np.isfinite(bc[0]) else np.nan,
            board_center_b=bc[1] if np.isfinite(bc[1]) else np.nan, center_sigma=cs, center_d2=cd,
            spread_ratio=sr, status=status, diagnosis='；'.join(reason) if reason else '正常'
        ))
    r = pd.DataFrame(rows)
    if not r.empty:
        tg = r.global_good.sum()
        r['valid_channel_contribution'] = r.global_good / tg if tg else 0.0
        return r.sort_values('display_name', ascending=True).reset_index(drop=True)
    return r

# ----------------- GUI 界面 -----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title(f'DSA ATE 测试板卡一致性分析工具 {APP_VERSION}')
        root.geometry('1680x1020')
        self.tests = {}; self.dataset = ''; self.board_models = {}; self.global_model = None
        self.global_input = pd.DataFrame(); self.global_fit = pd.DataFrame(); self.global_df = pd.DataFrame()
        self.stats = pd.DataFrame(); self.baseline = None
        self.fit_conf = 0.975; self.board_conf = 0.95; self.global_conf = 0.95; self.support = 0.85
        self.map = {}
        self.build()

    def build(self):
        left = ttk.Frame(self.root, padding=12)
        left.pack(side='left', fill='y')
        
        ttk.Label(left, text='1. 数据导入与导出', font=('微软雅黑', 11, 'bold')).pack(anchor='w')
        ttk.Button(left, text='📂 选择日志文件（可多选）', command=self.load).pack(fill='x', pady=2)
        self.status = ttk.Label(left, text='未加载数据', foreground='gray', wraplength=270)
        self.status.pack(anchor='w', pady=4)
        
        ttk.Button(left, text='📋 导出全部板卡汇总明细 CSV', command=self.export_details).pack(fill='x', pady=2)
        ttk.Button(left, text='📈 导出全板统计与诊断矩阵 CSV', command=self.export_matrix).pack(fill='x', pady=2)
        ttk.Button(left, text='📤 导出各板卡独立校准 CSV', command=lambda: export_csv(self.tests)).pack(fill='x', pady=(2, 8))
        
        ttk.Separator(left).pack(fill='x', pady=6)
        ttk.Label(left, text='2. 模型与参数设定', font=('微软雅黑', 11, 'bold')).pack(anchor='w')
        ttk.Label(left, text='测试项目 / 挡位').pack(anchor='w')
        self.ds = ttk.Combobox(left, state='readonly', width=32)
        self.ds.pack(fill='x', pady=(2, 6))
        self.ds.bind('<<ComboboxSelected>>', lambda e: self.fit_all())
        
        ttk.Label(left, text='模型拟合核心比例（改变最优中心 k,b）', font=('微软雅黑', 9, 'bold')).pack(anchor='w')
        self.sf = ttk.Scale(left, from_=0.70, to=0.999, value=self.fit_conf, orient='horizontal', length=275, command=self.fit_slider)
        self.sf.pack()
        self.lf = ttk.Label(left, text=f'拟合核心比例：{self.fit_conf * 100:.1f}%')
        self.lf.pack(anchor='w', pady=(1, 6))
        
        ttk.Label(left, text='板内判定置信度（板内异常筛选）', font=('微软雅黑', 9, 'bold')).pack(anchor='w')
        self.sb = ttk.Scale(left, from_=0.50, to=0.999, value=self.board_conf, orient='horizontal', length=275, command=self.board_slider)
        self.sb.pack()
        self.lb = ttk.Label(left, text=f'板内判定：{self.board_conf * 100:.1f}%')
        self.lb.pack(anchor='w', pady=(1, 6))
        
        ttk.Label(left, text='全局判定置信度（全局边界）', font=('微软雅黑', 9, 'bold')).pack(anchor='w')
        self.sg = ttk.Scale(left, from_=0.50, to=0.999, value=self.global_conf, orient='horizontal', length=275, command=self.global_slider)
        self.sg.pack()
        self.lg = ttk.Label(left, text=f'全局判定：{self.global_conf * 100:.1f}%')
        self.lg.pack(anchor='w', pady=(1, 8))
        
        ttk.Button(left, text='▶ 强制重新拟合全部模型', command=self.fit_all).pack(fill='x', pady=3)
        ttk.Separator(left).pack(fill='x', pady=6)
        
        ttk.Label(left, text='3. 黄金基线', font=('微软雅黑', 11, 'bold')).pack(anchor='w')
        ttk.Button(left, text='💾 保存当前全局模型为基线', command=self.save_baseline).pack(fill='x', pady=2)
        self.bs = ttk.Label(left, text='未加载基线', foreground='gray', wraplength=270)
        self.bs.pack(anchor='w', pady=4)
        
        ttk.Separator(left).pack(fill='x', pady=6)
        ttk.Label(left, text='4. 诊断摘要', font=('微软雅黑', 11, 'bold')).pack(anchor='w')
        self.info = tk.Text(left, width=38, height=22, font=('微软雅黑', 9), wrap='word')
        self.info.pack(fill='both', expand=True)
        self.info.tag_config('red', foreground='#d62728')
        self.info.tag_config('green', foreground='#2e8b57')
        
        nb = ttk.Notebook(self.root)
        nb.pack(side='right', fill='both', expand=True, padx=8, pady=8)
        self.t1 = ttk.Frame(nb); nb.add(self.t1, text=' 第一步：板内模型与通道清洗 ')
        self.t2 = ttk.Frame(nb); nb.add(self.t2, text=' 第二步：全局 3D + 2D 密度 ')
        self.t3 = ttk.Frame(nb); nb.add(self.t3, text=' 第三步：板卡健康度 ')
        self.build_tabs()

    def build_tabs(self):
        f = ttk.Frame(self.t1); f.pack(fill='x')
        ttk.Label(f, text='选择测试日志：').pack(side='left')
        self.bc = ttk.Combobox(f, state='readonly', width=78)
        self.bc.pack(side='left', padx=5)
        self.bc.bind('<<ComboboxSelected>>', lambda e: self.draw1())
        self.fig1 = Figure(figsize=(10, 7), dpi=100)
        self.a1 = self.fig1.add_subplot(111)
        self.c1 = FigureCanvasTkAgg(self.fig1, master=self.t1)
        self.c1.get_tk_widget().pack(fill='both', expand=True)
        
        self.fig2 = Figure(figsize=(12, 7.5), dpi=100)
        self.a3 = self.fig2.add_subplot(121, projection='3d')
        self.a2 = self.fig2.add_subplot(122)
        self.c2 = FigureCanvasTkAgg(self.fig2, master=self.t2)
        self.c2.get_tk_widget().pack(fill='both', expand=True)
        
        self.fig3 = Figure(figsize=(11, 7), dpi=100)
        self.a4 = self.fig3.add_subplot(111)
        self.c3 = FigureCanvasTkAgg(self.fig3, master=self.t3)
        self.c3.get_tk_widget().pack(fill='both', expand=True)

    def load(self):
        p = filedialog.askopenfilenames(title='选择日志文件（可多选）', filetypes=[('Log files', '*.log'), ('All files', '*.*')])
        if not p: return
        self.tests = load_logs(p)
        if not self.tests:
            messagebox.showwarning('提示', '未能成功读取到包含校准数据的日志文件。')
            return
        ds = sorted(set(k for t in self.tests.values() for k in t['datasets']))
        self.ds['values'] = ds
        self.ds.current(0)
        self.map = {t['display_name']: rid for rid, t in self.tests.items()}
        self.bc['values'] = sorted(self.map)
        self.bc.current(0)
        self.status.config(text=f'已加载 {len(self.tests)} 个测试日志\n包含 {len(ds)} 个分析挡位/项目', foreground='#2e8b57')
        self.fit_all()

    def export_details(self):
        if not self.tests:
            return messagebox.showwarning('提示', '请先导入日志数据！')
        p = filedialog.asksaveasfilename(title='保存全部板卡汇总明细 CSV', defaultextension='.csv',
                                         initialfile='全部板卡_校准明细汇总.csv', filetypes=[('CSV', '*.csv')])
        if not p: return
        export_consolidated_details_csv(self.tests, p)
        messagebox.showinfo('导出成功', f'全部板卡汇总明细数据已保存至：\n{p}')

    def export_matrix(self):
        if not self.tests:
            return messagebox.showwarning('提示', '请先导入日志数据！')
        p = filedialog.asksaveasfilename(title='保存全板统计与诊断矩阵 CSV', defaultextension='.csv',
                                         initialfile='全板卡_校准统计与诊断矩阵.csv', filetypes=[('CSV', '*.csv')])
        if not p: return
        export_matrix_statistics_csv(self.tests, p)
        messagebox.showinfo('导出成功', f'全板卡统计与诊断矩阵已保存至：\n{p}')

    def fit_slider(self, v):
        self.fit_conf = float(v)
        self.lf.config(text=f'拟合核心比例：{self.fit_conf * 100:.1f}%')
        self.fit_all()

    def board_slider(self, v):
        self.board_conf = float(v)
        self.lb.config(text=f'板内判定：{self.board_conf * 100:.1f}%')
        self.update_global()

    def global_slider(self, v):
        self.global_conf = float(v)
        self.lg.config(text=f'全局判定：{self.global_conf * 100:.1f}%')
        self.update_judgement()

    def fit_all(self):
        self.dataset = self.ds.get()
        if not self.dataset or not self.tests: return
        self.board_models = fit_board_models(self.tests, self.dataset, self.fit_conf, self.support)
        self.update_global()

    def update_global(self):
        if not self.board_models: return
        inp, _ = board_filter(self.board_models, self.board_conf)
        self.global_input = inp
        if inp.empty:
            self.global_model = None; self.global_df = pd.DataFrame(); self.stats = pd.DataFrame()
            self.draw_all(); return
        g = fit_global_balanced(inp, self.fit_conf, self.support)
        if not g:
            self.global_model = None; self.global_df = pd.DataFrame(); self.stats = pd.DataFrame()
            self.draw_all(); return
        self.global_model = g['model']; self.global_fit = g['fit_df']; self.global_target = g['target_per_run']
        self.update_judgement()

    def update_judgement(self):
        if self.global_model is None: return
        d, _ = apply_global(self.global_input, self.global_model, self.global_conf)
        if d.empty: return
        c = self.global_model['center']; cv = ensure_cov(self.global_model['cov'])
        d['global_center_k'] = c[0]; d['global_center_b'] = c[1]
        self.global_df = d
        self.stats = board_stats(d, c, cv)
        self.draw_all()

    def draw_all(self):
        self.draw1(); self.draw2(); self.draw3(); self.info_update()

    def draw1(self):
        self.a1.clear()
        name = self.bc.get(); rid = self.map.get(name)
        if not rid or rid not in self.board_models:
            self.c1.draw(); return
        m = self.board_models[rid]; d = m['df']; c = m['center']; med = m['raw_median']; cv = m['cov']
        co = cutoff(self.board_conf)
        ok = d['step1_is_ok'] = (d['step1_d2'] <= co) & np.isfinite(d['step1_d2'])
        good = d[ok]; bad = d[~ok]
        
        if len(good): self.a1.scatter(good.k, good.b, s=52, alpha=0.75, label='当前板内正常通道')
        if len(bad): self.a1.scatter(bad.k, bad.b, s=80, marker='x', linewidth=2, color='red', label='当前板内异常通道')
        for _, r in d.iterrows():
            self.a1.annotate(f"Ch{int(r.chn)}", (r.k, r.b), xytext=(3, 2), textcoords='offset points', fontsize=7.5, alpha=0.75)
        if np.all(np.isfinite(med)):
            self.a1.scatter(med[0], med[1], s=145, facecolors='none', edgecolors='black', linewidth=2, label='全体中位数参考', zorder=8)
        self.a1.scatter(c[0], c[1], marker='*', s=260, color='gold', edgecolor='black', label='最优鲁棒中心 (k,b)', zorder=9)
        ep = ellipse_params(c, cv, co)
        if ep:
            self.a1.add_patch(Ellipse((ep[0], ep[1]), ep[2], ep[3], angle=ep[4], fill=False, linestyle='--', linewidth=2.2, label=f'判定椭圆 {self.board_conf*100:.1f}%'))
            
        unit = d['b_unit'].iloc[0] if ('b_unit' in d.columns and not d.empty) else ''
        if np.all(np.isfinite(med)) and np.all(np.isfinite(c)):
            self.a1.text(0.02, 0.02,
                         f"样本中位数参考:\n  k = {med[0]:.7g}\n  b = {med[1]:.7g} {unit}\n\n"
                         f"鲁棒最优中心:\n  k = {c[0]:.7g}\n  b = {c[1]:.7g} {unit}\n\n"
                         f"中心差值:\n  Δk = {c[0]-med[0]:+.4g}\n  Δb = {c[1]-med[1]:+.4g} {unit}",
                         transform=self.a1.transAxes, va='bottom', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.45', alpha=0.88))
        self.a1.set_title(f'第一步：{name}\n总通道:{len(d)} | 正常:{len(good)} | 异常:{len(bad)}')
        self.a1.set_xlabel('k (Gain)')
        self.a1.set_ylabel(f'b (Offset / {unit})' if unit else 'b (Offset)')
        self.a1.grid(alpha=0.2)
        self.a1.legend(fontsize=8, loc='best')
        self.fig1.tight_layout()
        self.c1.draw()

    def draw2(self):
        self.a3.clear(); self.a2.clear()
        if self.global_model is None or self.global_df.empty:
            self.c2.draw(); return
        d = self.global_df; c = self.global_model['center']; cv = self.global_model['cov']
        ok = d[d.global_is_ok]; bad = d[~d.global_is_ok]
        ids = list(d.test_run_id.drop_duplicates()); z = {x: i for i, x in enumerate(ids)}
        
        if len(ok): self.a3.scatter(ok.k, ok.b, [z[x] for x in ok.test_run_id], s=15, alpha=0.6, label='全局正常')
        if len(bad): self.a3.scatter(bad.k, bad.b, [z[x] for x in bad.test_run_id], s=35, marker='x', color='red', alpha=0.9, label='全局异常')
        self.a3.set_xlabel('k'); self.a3.set_ylabel('b'); self.a3.set_zlabel('TestRun Index')
        self.a3.set_title('全局 3D 分层分布'); self.a3.legend(fontsize=8)
        
        xy = d[['k', 'b']].dropna().values
        if len(xy) >= 5:
            k0, k1 = xy[:, 0].min(), xy[:, 0].max()
            b0, b1 = xy[:, 1].min(), xy[:, 1].max()
            pk = max((k1 - k0) * 0.25, 0.001); pb = max((b1 - b0) * 0.25, 0.001)
            gx, gy = np.meshgrid(np.linspace(k0 - pk, k1 + pk, 120), np.linspace(b0 - pb, b1 + pb, 120))
            grid = np.vstack([gx.ravel(), gy.ravel()])
            try:
                if np.linalg.matrix_rank(np.cov(xy, rowvar=False)) >= 2:
                    kd = gaussian_kde(xy.T)(grid).reshape(gx.shape)
                    kd /= kd.max() if kd.max() > 0 else 1
                    self.a2.contourf(gx, gy, kd, levels=18, cmap='Blues', alpha=0.42)
            except Exception: pass
            d2 = md2(grid.T, c, cv).reshape(gx.shape)
            self.a2.contour(gx, gy, d2, levels=[cutoff(self.global_conf)], linestyles='--', linewidths=2)
            
        if len(ok): self.a2.scatter(ok.k, ok.b, s=16, alpha=0.65, label='全局正常')
        if len(bad): self.a2.scatter(bad.k, bad.b, s=35, marker='x', color='red', label='全局异常')
        self.a2.scatter(c[0], c[1], marker='*', s=260, color='gold', edgecolor='black', label='全局最优中心')
        ep = ellipse_params(c, cv, cutoff(self.global_conf))
        if ep:
            self.a2.add_patch(Ellipse((ep[0], ep[1]), ep[2], ep[3], angle=ep[4], fill=False, linestyle='--', linewidth=2))
            
        self.a2.set_title(f'全局核密度估计 (KDE) 与判定椭圆 ({self.global_conf*100:.1f}%)')
        self.a2.set_xlabel('k'); self.a2.set_ylabel('b')
        self.a2.grid(alpha=0.2); self.a2.legend(fontsize=8)
        self.fig2.tight_layout()
        self.c2.draw()

    def draw3(self):
        self.a4.clear()
        if self.stats.empty:
            self.c3.draw(); return
        s = self.stats.sort_values('display_name', ascending=True).reset_index(drop=True)
        bars = self.a4.bar(np.arange(len(s)), s.survival_rate * 100, alpha=0.85)
        for i, (_, r) in enumerate(s.iterrows()):
            if r.status == 'FAIL': bars[i].set_hatch('///')
            elif r.status == 'WARN': bars[i].set_hatch('//')
            self.a4.text(i, r.survival_rate * 100 + 1, f"{r.survival_rate * 100:.1f}%", ha='center', fontsize=8)
            
        self.a4.axhline(BOARD_SURVIVAL_PASS * 100, linestyle='--', color='green', linewidth=1, label=f'Pass ({BOARD_SURVIVAL_PASS*100:.0f}%)')
        self.a4.axhline(BOARD_SURVIVAL_WARN * 100, linestyle=':', color='red', linewidth=1, label=f'Warn ({BOARD_SURVIVAL_WARN*100:.0f}%)')
        self.a4.set_xticks(np.arange(len(s)))
        # 使用完整的日志显示名称（如 No1 BP-9007 ...）并倾斜展示
        labels = [r.display_name for _, r in s.iterrows()]
        self.a4.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        self.a4.set_ylim(0, 105)
        self.a4.set_ylabel('全局通道存活率 (%)')
        self.a4.set_title('板卡全局一致性健康度综合排名')
        self.a4.grid(axis='y', alpha=0.2)
        self.a4.legend(loc='lower right', fontsize=8)
        self.fig3.tight_layout()
        self.fig3.subplots_adjust(bottom=0.25)
        self.c3.draw()

    def info_update(self):
        self.info.config(state='normal')
        self.info.delete('1.0', 'end')
        if self.global_model is None or self.global_df.empty:
            self.info.insert('1.0', '暂无全局分析数据。')
            self.info.config(state='disabled')
            return
        c = self.global_model['center']
        total = len(self.global_df)
        gg = int(self.global_df.global_is_ok.sum())
        s1 = int(self.global_df.step1_is_ok.sum())
        unit = self.global_df['b_unit'].iloc[0] if 'b_unit' in self.global_df.columns else ''
        
        txt = (
            f"【模型状态】\n"
            f"板内算法: RMCD + 鲁棒中心细化\n"
            f"全局算法: Balanced-RMCD\n\n"
            f"【全局最优中心 (k,b)】\n"
            f"k0 (Gain)   = {c[0]:.8g}\n"
            f"b0 (Offset) = {c[1]:.8g} {unit}\n\n"
            f"【统计概要】\n"
            f"板内过滤后输入 = {s1}\n"
            f"全局判定正常数 = {gg}\n"
            f"总通道存活率   = {gg/total*100:.2f}%\n"
        )
        self.info.insert('1.0', txt)
        if not self.stats.empty:
            for _, r in self.stats[self.stats.status == 'FAIL'].head(6).iterrows():
                self.info.insert('end', f"\nFAIL {r.display_name}:\n  存活率: {r.survival_rate*100:.1f}% | {r.diagnosis}", 'red')
            for _, r in self.stats[self.stats.status == 'WARN'].head(6).iterrows():
                self.info.insert('end', f"\nWARN {r.display_name}:\n  存活率: {r.survival_rate*100:.1f}% | {r.diagnosis}", 'red')
        self.info.config(state='disabled')

    def save_baseline(self):
        if self.global_model is None:
            return messagebox.showwarning('提示', '请先选择测试项目并完成全局拟合')
        c = self.global_model['center']
        path = filedialog.asksaveasfilename(defaultextension='.json', initialfile=f'RMCD_GoldenBaseline_{self.dataset}.json', filetypes=[('JSON', '*.json')])
        if not path: return
        data = {
            'app_version': APP_VERSION, 'dataset': self.dataset, 'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'k_center': float(c[0]), 'b_center': float(c[1]),
            'b_unit': str(self.global_df['b_unit'].iloc[0] if 'b_unit' in self.global_df.columns else ''),
            'covariance_matrix': ensure_cov(self.global_model['cov']).tolist(),
            'model_input_count': int(len(self.global_input)), 'balanced_fit_count': int(len(self.global_fit))
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.baseline = data
        self.bs.config(text=f'基线已保存\n{data["save_time"]}', foreground='#2e8b57')
        messagebox.showinfo('成功', f'基线模型已保存至：\n{path}')

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
