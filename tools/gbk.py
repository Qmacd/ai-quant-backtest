def convert_csv_to_gbk(input_file, output_file=None):
    """
    将CSV文件转换为GBK编码
    
    参数:
        input_file (str): 输入CSV文件的路径
        output_file (str): 输出文件的路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file
        
    try:
        # 尝试用不同的编码方式读取文件
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
                
        if content is None:
            raise Exception("无法识别文件编码")
            
        # 用GBK编码保存文件
        with open(output_file, 'w', encoding='gbk', errors='ignore') as f:
            f.write(content)
            
        print(f"文件已成功转换为GBK编码：{output_file}")
        
    except Exception as e:
        print(f"转换过程中出现错误：{str(e)}")

if __name__ == "__main__":
    convert_csv_to_gbk("signal_info.csv")
