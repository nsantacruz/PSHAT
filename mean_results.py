import re

results_root = "results_txt"
input = "land-id.txt"

with open("{}/{}".format(results_root, input), "rb") as f:

    num = 0
    total = 0.0
    for line in f:
        pos_prec = float(re.findall(r"lang_prec: (.+?)$", line)[0])
        print pos_prec
        total += pos_prec
        num += 1

    print total / num
