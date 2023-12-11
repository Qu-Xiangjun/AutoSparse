import sys
import re

if __name__ == "__main__":
    f = open(sys.argv[1]) # ./taco_kernel.c
    s = f.read()
    f.close()

    f = open(sys.argv[1], "w")
    apos = re.findall("A_vals\[(.*?)\]", s)[0] # 找到 A_vals[] 括号中的内容
    # 字符串 C_vals[] 中括号中的内容替换为之前获取的 apos 变量的值c
    substitute_C_vals = re.sub("C_vals\[(.*?)\]", 'C_vals[{}]'.format(apos), s) 
    # 将形如 C_vals[] = 0.0; 的内容用空字符串替换，即注释掉这样的行。
    commentout_zero = re.sub("C_vals\[.*\] = 0.0;", "", substitute_C_vals)
    f.write(commentout_zero)
    f.close()

