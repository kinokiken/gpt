"""Prepare a colour mask from the local Treponema illustration."""
from experiment_common import *
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

def main():
    image = cv2.imread(str(ROOT/'data/raw/treponema_primary.jpg'))
    if image is None:
        raise FileNotFoundError('data/raw/treponema_primary.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (image[:, :, 1] < 55) & (hsv[:, :, 1] > 100)
    mask = remove_small_objects(mask, max_size=119)
    mask = remove_small_holes(mask, max_size=149)
    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_CLOSE,
                          np.ones((3, 3), np.uint8))
    skeleton = skeletonize(mask > 0)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    cv2.imwrite(str(ROOT/'results/primary_mask.png'), mask * 255)
    cv2.imwrite(str(ROOT/'results/primary_skeleton.png'), skeleton.astype('uint8') * 255)
    save('primary_preparation.json', {
        'shape': list(image.shape[:2]),
        'mask_fraction': float(mask.mean()),
        'components_over_450': int((stats[1:, 4] > 450).sum())})

if __name__ == '__main__':
    main()
