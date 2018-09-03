# encoding: utf8

bad = ""
lines = []

with open("subj.txt") as f:
    for line in f:
        line = (line
            .strip()
            .replace(bad, "")
            .replace("[", "")
            .replace("]", "")
        )
        lines.append(line)


with open("subj.txt", "w") as f:
    for line in lines:
        print(line, file=f)
