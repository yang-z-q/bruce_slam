#!/usr/bin/env python

def test_file_write():
    output_txt_path = "../test_messages.txt"
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            # 写入一些测试数据
            txt_file.write("测试数据1\n")
            txt_file.write("测试数据2\n")
            txt_file.write("-" * 80 + "\n")
            txt_file.flush()
        print(f"测试数据已成功写入 {output_txt_path}")
    except Exception as e:
        print(f"写入文件时发生错误: {str(e)}")

if __name__ == "__main__":
    test_file_write() 