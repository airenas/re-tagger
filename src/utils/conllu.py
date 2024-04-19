import sys


def extract_tag(l):
    v = l.partition("Multext=")
    if v[1]:
        return v[2]
    return v[0]


class Connlu:
    def __init__(self):
        self.lines = []

    def add(self, line):
        self.lines.append(line)

    def sentence(self):
        return ' '.join(self.words())

    def words(self):
        return map(lambda l: (l.split("\t")[1]), filter(lambda l: (not l.startswith('#')), self.lines))

    def main_forms(self):
        return map(lambda l: (l.split("\t")[2]), filter(lambda l: (not l.startswith('#')), self.lines))

    def tags(self):
        for l in self.lines:
            if not l.startswith('#'):
                colls = l.split("\t")
                if len(colls) < 10:
                    raise RuntimeError(f"wrong line: {l}\n{colls}")
        return map(lambda l: extract_tag(l),
                   map(lambda l: (l.split("\t")[9]), filter(lambda l: (not l.startswith('#')), self.lines)))


class ConlluReader:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, path):
        self.path = path
        self.read = 0

    def __iter__(self):
        conllu = Connlu()
        for line in self.fd:
            line = line.strip()
            if not line:
                if len(conllu.lines) > 0:
                    yield conllu
                conllu = Connlu()
            else:
                conllu.add(line)
        if len(conllu.lines) > 0:
            yield conllu

    def __enter__(self):
        self.fd = open(self.path, 'r')
        print("Opened %s" % self.path, file=sys.stderr)
        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()
