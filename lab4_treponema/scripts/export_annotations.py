"""Export the primary illustration's mask geometry as CVAT boxes and points."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--image', default='data/raw/treponema_primary.jpg')
    parser.add_argument('--mask', default='results/primary_mask.png')
    parser.add_argument('--out', default='results/cvat/annotations.xml')
    args = parser.parse_args()
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape:
        raise ValueError('Readable image and mask of equal dimensions required')
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype('uint8'))
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'
    node = ET.SubElement(root, 'image', id='0', name=Path(args.image).name,
                         width=str(image.shape[1]), height=str(image.shape[0]))
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area <= 450:
            continue
        yy, xx = np.nonzero(labels == label)
        points = np.column_stack([xx, yy]).astype(float)
        center = points.mean(0)
        _, _, directions = np.linalg.svd(points - center, full_matrices=False)
        axis = directions[0]
        projection = (points - center) @ axis
        ends = center + np.array([projection.min(), projection.max()])[:, None] * axis
        common = {'occluded': '0', 'source': 'auto', 'group_id': str(label)}
        ET.SubElement(node, 'box', label='treponema', xtl=str(x), ytl=str(y),
                      xbr=str(x + w - 1), ybr=str(y + h - 1), **common)
        for name, point in zip(['start', 'center', 'end'], [ends[0], center, ends[1]]):
            point = np.clip(point, [0, 0], [image.shape[1]-1, image.shape[0]-1])
            ET.SubElement(node, 'points', label=name,
                          points=f'{point[0]:.3f},{point[1]:.3f}', **common)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
    print('Saved', target, '; create matching labels in CVAT before import.')

if __name__ == '__main__':
    main()
