out = []
with open("conll14st/official-2014.combined.m2", 'r') as f:
    for line in f.readlines():
        if line[0] == 'S':
            out.append(line[2:])
with open("conll14st/combined.tok", 'w') as fout:
    fout.writelines(out)