"""colab_cell.py -- first code cell of the NN notebooks: on Google Colab it clones the repository and moves into
the notebook's own folder, so the relative paths (../data/clean/, ../common, ...) work as on a local copy."""
import nbformat

HEAD = {"en": '# Google Colab: copy the course repository, so that the data and the saved results are found', "ro": '# Google Colab: copiem repo-ul cursului, ca datele și rezultatele salvate să fie găsite'}
MARK = "google.colab"


def colab_code(lang="en", subdir="notebooks"):
    return (HEAD["ro" if lang == "ro" else "en"] + "\nimport os, subprocess, sys\n"
            "if 'google.colab' in sys.modules and not os.getcwd().startswith('/content/NEURAL_BIZ'):\n"
            "    if not os.path.isdir('/content/NEURAL_BIZ'):\n"
            "        subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/danpele/NEURAL_BIZ.git', '/content/NEURAL_BIZ'], check=True)\n"
            f"    os.chdir('/content/NEURAL_BIZ/{subdir}')")


def add_colab_cell(nb, lang="en", subdir="notebooks"):
    """Insert (or refresh) the Colab cell before the first code cell."""
    nb.cells = [c for c in nb.cells if not (c.cell_type == "code" and MARK in c.source and "git', 'clone" in c.source)]
    i = next((k for k, c in enumerate(nb.cells) if c.cell_type == "code"), len(nb.cells))
    nb.cells.insert(i, nbformat.v4.new_code_cell(colab_code(lang, subdir)))
    return nb
