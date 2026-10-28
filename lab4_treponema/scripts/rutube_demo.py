"""Illustrate a hypothetical search result without accessing RuTube."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results'
    out.mkdir(exist_ok=True)
    rows = [{'seconds': 18, 'cosine': .31}, {'seconds': 36, 'cosine': .29},
            {'seconds': 62, 'cosine': .27}]
    payload = {'status': 'simulated_not_measured', 'source_url': None,
               'query': 'spiral bacteria under a microscope', 'hits': rows}
    (out / 'rutube_demo.json').write_text(json.dumps(payload, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([f"{r['seconds']} с" for r in rows], [r['cosine'] for r in rows], color='#6243a5')
    ax.set_ylim(0, .4)
    ax.set_ylabel('Условная cosine similarity')
    ax.set_xlabel('Придуманные временные отметки')
    ax.set_title('Демонстрационный пример выдачи: данные смоделированы')
    fig.text(.5, .01, 'RuTube не запускался. Это схема, а не результат измерений.', ha='center')
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(out / 'rutube_demo.png', dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    main()
