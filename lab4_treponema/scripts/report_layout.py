"""Arrange explanations and complete source files inside their report sections."""
from pathlib import Path
import re

def arrange(blocks, root):
    sections = {}
    extra = {i: [] for i in range(1, 12)}
    sources = []
    destination = None
    maps = {
        'Б': {1:2, 2:3, 3:3, 4:1},
        'В': {1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:5, 8:6, 9:7, 10:9},
        'Г': {1:2, 2:7, 3:3, 4:3, 5:3, 6:3, 7:4, 8:5, 9:7, 10:7},
        'Д': {1:1, 2:2, 3:7, 4:1, 5:1},
    }
    for block in blocks:
        text = block.get('text', '')
        if block['kind'] == 'h':
            if block['level'] == 1:
                match = re.match(r'^(\d+)\. ', text)
                if match:
                    number = int(match[1])
                    sections[number] = [block]
                    destination = sections[number]
                elif text.startswith('Список'):
                    destination = sources
                    destination.append(block)
                else:
                    destination = None
                continue
            match = re.match(r'^([БВГД])\.(\d+)\. (.+)', text)
            if match:
                letter, number, title = match.groups()
                destination = extra[maps[letter][int(number)]]
                destination.append({'kind':'h', 'level':2, 'text':title})
                continue
        if destination is None or block['kind'] == 'code' or text.startswith('Листинг '):
            continue
        if block['kind'] == 'p':
            block['text'] = block['text'].replace('условия S < 55 для зелёного канала', 'условия G < 55 для зелёного канала')
        destination.append(block)

    programs = {
        1: ['fetch_open_data.py', 'verify_checkpoint.py'],
        2: ['prepare_and_visualize.py'],
        3: ['analyze_video.py', 'image_controls.py'],
        4: ['retrieval.py'], 5: ['train_metric.py'], 6: ['zero_shot.py'],
        7: ['video_index.py', 'app.py', 'rutube_download.py'],
        8: ['export_annotations.py'], 9: ['log_run.py'], 10: ['pose_cycle.py'],
    }
    notes = {
        3: 'Приведённые ниже старые числовые таблицы и диаграммы трекинга и сверхразрешения воспроизводят константы из prepare_and_visualize.py. Они не подтверждены журналом измерений на t07.ogv. Полные программы этого пункта сохраняют новые фактические наблюдения при запуске.',
        4: 'Значение Precision@3 на прежнем графике задано константой. Размеченная контрольная выдача, подтверждающая это значение для данного набора, отсутствует. Программа ниже сохраняет реальные сходства; оценка релевантности выполняется отдельно.',
        5: 'Значения 0,990 и 1,000 на прежнем графике заданы вручную и не являются подтверждённым результатом обучения на Treponema pallidum. Программа ниже требует файл векторов и отдельный JSON с метками обучающих примеров.',
        6: 'Числа на прежней диаграмме заданы в скрипте построения рисунка, а не извлечены из журналов моделей. Они приведены как иллюстративная сводка. Полная программа ниже реализует YOLO-World; остальные модели описаны методически.',
        9: 'Прежние значения Macro-F1 в диаграмме не подтверждены сохранённым запуском MLflow для этого набора. Полная программа принимает измеренные метрики из файла и записывает их без встроенных числовых результатов.',
        10: 'Числа в прежней таблице сравнения YOLO заданы в скрипте визуализации. Запуски обучения, подтверждающие эти показатели на Treponema pallidum, не представлены. Для программы ниже сначала необходимо подготовить размеченный набор и YAML.',
    }
    result = []
    for number, section in sections.items():
        if number in notes:
            section.insert(1, {'kind':'p', 'text':notes[number]})
            for block in section:
                if block['kind'] == 'table':
                    block['headers'] = [v.replace('Фактический результат', 'Значение прежнего примера')
                                         .replace('Контрольный результат', 'Иллюстративное значение')
                                         .replace('Получено', 'Значение примера') for v in block['headers']]
                if block['kind'] == 'figure':
                    block['caption'] += ' Числа из прежнего примера; измерениями этого запуска не подтверждены.'
            section = [b for b in section if not (b['kind']=='p' and
                       b.get('text','').startswith('YOLO11s-pose показала'))]
        if number == 11:
            section = [section[0], {'kind':'p', 'text':
                'Выполнена обработка основного изображения: получены цветовая маска, скелет и геометрические ориентиры. Подготовлены полные программы для последующих опытов и поиска по видео. Для сравнения моделей необходимо получить журналы их запусков на подготовленных данных. Демонстрационный пример RuTube содержит условные значения и не является сетевым экспериментом.'}]
        result.extend(section)
        result.extend(extra[number])
        if number in programs:
            result.append({'kind':'h', 'level':2, 'text':'Полные программы и входные данные'})
            result.append({'kind':'p', 'text':
                'Ниже приведены файлы целиком, включая импорты и точку входа. Рабочий каталог — lab4_treponema; общие модули находятся в scripts. Новые программы расчётов включены для воспроизведения; их добавление не означает, что вычислительный эксперимент уже выполнен.'})
            for index, filename in enumerate(programs[number], 1):
                result.append({'kind':'p', 'text':f'Листинг {number}.{index} — scripts/{filename}, полный файл.'})
                result.append({'kind':'code', 'text':(root/'scripts'/filename).read_text(encoding='utf-8')})
        if number == 7:
            result.extend([
                {'kind':'h','level':2,'text':'Демонстрационный пример выдачи RuTube'},
                {'kind':'p','text':'Условный запрос: spiral bacteria under a microscope. Все значения ниже придуманы для пояснения формата выдачи. Они не относятся к реальному ролику; URL и название не назначены. RuTube не запускался.'},
                {'kind':'table','headers':['Условное место','Время, с','Условная cosine similarity'],
                 'rows':[['1','18','0,31'],['2','36','0,29'],['3','62','0,27']], 'widths':[.3,.3,.4]},
                {'kind':'figure','path':str(root/'results/rutube_demo.png'),
                 'caption':'Схема теоретической выдачи. Данные смоделированы, сетевой запуск отсутствует.'},
                {'kind':'p','text':'Листинг 7.4 — scripts/rutube_demo.py. Полная программа построения демонстрационной схемы, без сетевых запросов.'},
                {'kind':'code','text':(root/'scripts/rutube_demo.py').read_text(encoding='utf-8')},
            ])
    # Consecutive subsection numbering, reflecting the new placement.
    section_number = 0
    counter = 0
    for block in result:
        if block['kind'] == 'h':
            if block['level'] == 1:
                section_number += 1
                counter = 0
            else:
                counter += 1
                title = re.sub(r'^\d+\.\d+\.\s*', '', block['text'])
                block['text'] = f'{section_number}.{counter}. {title}'
    return result + sources
