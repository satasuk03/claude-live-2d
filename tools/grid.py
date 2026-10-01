import sys
from PIL import Image, ImageDraw
# grid.py img out x0 y0 x1 y1 outW step
p,out,x0,y0,x1,y1,ow,step=sys.argv[1],sys.argv[2],*map(int,sys.argv[3:9])
im=Image.open(p).convert('RGB').crop((x0,y0,x1,y1)); s=ow/(x1-x0)
im=im.resize((ow,int((y1-y0)*s))); d=ImageDraw.Draw(im)
for x in range((x0//step+1)*step, x1, step):
    X=(x-x0)*s; d.line([(X,0),(X,im.height)],fill=(255,0,0) if x%(step*5)==0 else (0,160,255),width=1); d.text((X+2,2),str(x),fill=(255,0,0))
for y in range((y0//step+1)*step, y1, step):
    Y=(y-y0)*s; d.line([(0,Y),(im.width,Y)],fill=(255,0,0) if y%(step*5)==0 else (0,160,255),width=1); d.text((2,Y+2),str(y),fill=(255,0,0))
im.save(out,quality=90)
