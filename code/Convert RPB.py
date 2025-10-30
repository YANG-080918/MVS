import os

rpc_txt_dir = r"D:\MVS3DM\rpc"
rpb_dir = r"D:\MVS3DM\convert rpc"

def find_matching_txt(rpb_filename):
    base_name = rpb_filename.replace('.RPB', '')
    for txt_filename in os.listdir(rpc_txt_dir):
        if base_name in txt_filename:
            return os.path.join(rpc_txt_dir, txt_filename)
    return None

def extract_numbers_from_txt(txt_filepath):
    with open(txt_filepath, 'r') as txt_file:
        content = txt_file.read().strip()
    numbers = [float(x.strip()) for x in content.split(',')]
    return numbers[-2], numbers[-1]

def modify_rpb_file(rpb_filepath, samp_offset_correction, line_offset_correction):
    with open(rpb_filepath, 'r') as rpb_file:
        lines = rpb_file.readlines()
    for i, line in enumerate(lines):
        if "sampOffset" in line:
            samp_offset_str = line.split('=')[1].strip().replace(';', '')
            try:
                samp_offset = float(samp_offset_str)
                new_samp_offset = samp_offset - samp_offset_correction
                lines[i] = f"                sampOffset = {new_samp_offset:.6f};\n"
            except ValueError:
                print(f"错误: 无法转换sampOffset的值 '{samp_offset_str}' 为浮点数")
        if "lineOffset" in line:
            line_offset_str = line.split('=')[1].strip().replace(';', '')
            try:
                line_offset = float(line_offset_str)
                new_line_offset = line_offset - line_offset_correction
                lines[i] = f"                lineOffset = {new_line_offset:.6f};\n"
            except ValueError:
                print(f"错误: 无法转换lineOffset的值 '{line_offset_str}' 为浮点数")
    with open(rpb_filepath, 'w') as rpb_file:
        rpb_file.writelines(lines)

for rpb_filename in os.listdir(rpb_dir):
    if rpb_filename.endswith('.RPB'):
        rpb_filepath = os.path.join(rpb_dir, rpb_filename)
        matching_txt_filepath = find_matching_txt(rpb_filename)
        
        if matching_txt_filepath:
            try:
                samp_offset_correction, line_offset_correction = extract_numbers_from_txt(matching_txt_filepath)
                modify_rpb_file(rpb_filepath, samp_offset_correction, line_offset_correction)
                print(f"已处理: {rpb_filename}")
            except Exception as e:
                print(f"处理文件 {rpb_filename} 时出错: {e}")
        else:
            print(f"未找到匹配的txt文件，跳过: {rpb_filename}")





