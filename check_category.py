from pycocotools.coco import COCO

coco = COCO("./data/annotations/train.json")
print("All categories:", coco.getCatIds())  # Должно [1,2,3] + другие

bars_imgs = coco.getImgIds(catIds=[1])
lines_imgs = coco.getImgIds(catIds=[2])
pies_imgs = coco.getImgIds(catIds=[3])
cat_4 = coco.getImgIds(catIds=[4])

print(f"Images with bars (id=1): {len(bars_imgs)}")
print(f"Images with bars (id=1): {len(bars_imgs)}")
print(f"Images with lines (id=2): {len(lines_imgs)}")
print(f"Images with (id=4): {len(cat_4)}")
print(f"Images with bars OR lines: {len(coco.getImgIds(catIds=[1,2]))}")