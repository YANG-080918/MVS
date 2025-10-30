import re
from pathlib import Path

def _read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

def _write_text(path, s):
    Path(path).write_text(s, encoding="utf-8")

def _extract_token(s):
    s = s.strip()
    s = re.sub(r'[;,)]$', '', s).strip()
    return s

def _parse_rpb(path):
    lines = _read_text(path)
    inside = False
    kv = {}
    coefs = {"lineNumCoef": [], "lineDenCoef": [], "sampNumCoef": [], "sampDenCoef": []}
    current_block = None

    for raw in lines:
        line = raw.strip()

        if line.startswith("BEGIN_GROUP") and "IMAGE" in line:
            inside = True
            continue
        if line.startswith("END_GROUP") and "IMAGE" in line:
            inside = False
            continue
        if not inside:
            continue

        m = re.match(
            r'^(lineOffset|sampOffset|latOffset|longOffset|heightOffset|'
            r'lineScale|sampScale|latScale|longScale|heightScale)\s*=\s*(.+)$', line)
        if m:
            k, v = m.group(1), _extract_token(m.group(2))
            kv[k] = v
            continue

        for key in ("lineNumCoef", "lineDenCoef", "sampNumCoef", "sampDenCoef"):
            if line.startswith(f"{key}"):
                current_block = key
                continue

        if current_block:
            if line.endswith(");"):
                val = _extract_token(line[:-2])
                if val:
                    val = val.rstrip(',').strip()
                    if val:
                        coefs[current_block].append(val)
                current_block = None
            else:
                val = _extract_token(line).rstrip(',').strip()
                if val:
                    coefs[current_block].append(val)

    for k in coefs:
        if len(coefs[k]) != 20:
            raise ValueError(f"RPB parsing error: {k} expects 20 coefficients, got {len(coefs[k])}")
    return kv, coefs

def _write_rpc(path, kv, coefs):
    def g(k): 
        if k not in kv:
            raise KeyError(f"Missing key in RPB: {k}")
        return kv[k]

    out = []
    out.append(f"LINE_OFF: {g('lineOffset')}  pixels")
    out.append(f"SAMP_OFF: {g('sampOffset')}  pixels")
    out.append(f"LAT_OFF: {g('latOffset')}   degrees")
    out.append(f"LONG_OFF: {g('longOffset')}   degrees")
    out.append(f"HEIGHT_OFF: {g('heightOffset')}   meters")
    out.append(f"LINE_SCALE: {g('lineScale')}  pixels")
    out.append(f"SAMP_SCALE: {g('sampScale')}  pixels")
    out.append(f"LAT_SCALE: {g('latScale')}   degrees")
    out.append(f"LONG_SCALE: {g('longScale')}   degrees")
    out.append(f"HEIGHT_SCALE: {g('heightScale')}   meters")

    for i, v in enumerate(coefs["lineNumCoef"], 1):
        out.append(f"LINE_NUM_COEFF_{i}: {v}")
    for i, v in enumerate(coefs["lineDenCoef"], 1):
        out.append(f"LINE_DEN_COEFF_{i}: {v}")
    for i, v in enumerate(coefs["sampNumCoef"], 1):
        out.append(f"SAMP_NUM_COEFF_{i}: {v}")
    for i, v in enumerate(coefs["sampDenCoef"], 1):
        out.append(f"SAMP_DEN_COEFF_{i}: {v}")

    _write_text(path, "\n".join(out) + "\n")

def _parse_rpc(path):
    lines = _read_text(path)
    kv = {}
    coef_map = {
        "LINE_NUM_COEFF": [],
        "LINE_DEN_COEFF": [],
        "SAMP_NUM_COEFF": [],
        "SAMP_DEN_COEFF": [],
    }

    for raw in lines:
        line = raw.strip()
        m = re.match(r'^([A-Z_]+):\s*([^\s]+)', line)
        if m:
            key, val = m.group(1), m.group(2)
            if key in ("LINE_OFF", "SAMP_OFF", "LAT_OFF", "LONG_OFF", "HEIGHT_OFF",
                       "LINE_SCALE", "SAMP_SCALE", "LAT_SCALE", "LONG_SCALE", "HEIGHT_SCALE"):
                kv[key] = _extract_token(val)
                continue

        m2 = re.match(r'^(LINE_NUM_COEFF|LINE_DEN_COEFF|SAMP_NUM_COEFF|SAMP_DEN_COEFF)_(\d+):\s*(.+)$', line)
        if m2:
            base, idx, val = m2.group(1), int(m2.group(2)), _extract_token(m2.group(3))
            coef_map[base].append((idx, val))

    for k in coef_map:
        coef_map[k].sort(key=lambda x: x[0])
        if len(coef_map[k]) != 20:
            raise ValueError(f"RPC parsing error: {k} expects 20 coefficients, got {len(coef_map[k])}")

    return kv, {k: [v for _, v in coef_map[k]] for k in coef_map}

def _write_rpb(path, kv_rpc, coefs_rpc):
    def r(name):
        if name not in kv_rpc:
            raise KeyError(f"Missing key in RPC: {name}")
        return kv_rpc[name]

    map_keys = {
        "lineOffset":  r("LINE_OFF"),
        "sampOffset":  r("SAMP_OFF"),
        "latOffset":   r("LAT_OFF"),
        "longOffset":  r("LONG_OFF"),
        "heightOffset":r("HEIGHT_OFF"),
        "lineScale":   r("LINE_SCALE"),
        "sampScale":   r("SAMP_SCALE"),
        "latScale":    r("LAT_SCALE"),
        "longScale":   r("LONG_SCALE"),
        "heightScale": r("HEIGHT_SCALE"),
    }

    out = []
    out.append('satId = "XXX";')
    out.append('bandId = "XXX";')
    out.append('SpecId = "XXX";')
    out.append('BEGIN_GROUP = IMAGE')
    out.append('\terrBias =   1.0;')
    out.append('\terrRand =    0.0;')

    out.append(f'\tlineOffset = {map_keys["lineOffset"]}')
    out.append(f'\tsampOffset = {map_keys["sampOffset"]}')
    out.append(f'\tlatOffset =  {map_keys["latOffset"]}')
    out.append(f'\tlongOffset =  {map_keys["longOffset"]}')
    out.append(f'\theightOffset =  {map_keys["heightOffset"]}')
    out.append(f'\tlineScale =  {map_keys["lineScale"]}')
    out.append(f'\tsampScale =  {map_keys["sampScale"]}')
    out.append(f'\tlatScale = {map_keys["latScale"]}')
    out.append(f'\tlongScale =  {map_keys["longScale"]}')
    out.append(f'\theightScale = {map_keys["heightScale"]}')

    def block(name, arr):
        out.append(f'\t{name} = (')
        for i, v in enumerate(arr, 1):
            sep = ',' if i < len(arr) else ')'
            if i < len(arr):
                out.append(f'\t\t\t{v},')
            else:
                out.append(f'\t\t\t{v})' + ';')

    block('lineNumCoef',  coefs_rpc['LINE_NUM_COEFF'])
    block('lineDenCoef',  coefs_rpc['LINE_DEN_COEFF'])
    block('sampNumCoef',  coefs_rpc['SAMP_NUM_COEFF'])
    block('sampDenCoef',  coefs_rpc['SAMP_DEN_COEFF'])

    out.append('END_GROUP = IMAGE')
    out.append('END;')

    _write_text(path, "\n".join(out) + "\n")

def rpb2rpc(f_in_name, f_out_name, transform_type):
    t = int(transform_type)
    if t == 1:
        kv, coefs = _parse_rpb(f_in_name)
        _write_rpc(f_out_name, kv, coefs)
    elif t == 2:
        kv_rpc, coefs_rpc = _parse_rpc(f_in_name)
        _write_rpb(f_out_name, kv_rpc, coefs_rpc)
    else:
        raise ValueError("transform_type must be 1 (RPB->RPC) or 2 (RPC->RPB)")
