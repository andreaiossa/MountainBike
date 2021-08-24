from urllib import request
from urllib.request import urlretrieve

url = 'https://pjreddie.com/media/files/yolov3.weights'

file_name = url.split('/')[-1]
u = request.urlopen(url)
f = open(f"./{file_name}", 'wb')
meta = u.info()
file_size = int(meta.get_all("Content-Length")[0])

print(f"Downloading: {file_name} Bytes: {file_size}")

file_size_dl = 0
block_sz = 8192
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
    status = status + chr(8) * (len(status) + 1)
    print(status),

f.close()