# coding=utf-8

from urllib import request

# downloader
def download(url, encoding='gbk', flag=1):
    response = request.urlopen(url)
    content = response.read()
    if flag == 1:  # 文本
        return content.decode(encoding)
    return content
    

def main():
    url = 'http://www.quanshuwang.com/book/9/9055/9674264.html'
    html = download(url, 'gbk')
    print(html)

if __name__ == '__main__':
    main()
