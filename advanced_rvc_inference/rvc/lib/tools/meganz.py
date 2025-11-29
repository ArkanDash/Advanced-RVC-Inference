import os
import re
import sys
import json
import tqdm
import codecs
import random
import base64
import struct
import shutil
import requests
import tempfile

from Crypto.Cipher import AES
from Crypto.Util import Counter

sys.path.append(os.getcwd())

from main.app.variables import translations

def makebyte(x):
    return codecs.latin_1_encode(x)[0]

def a32_to_str(a):
    return struct.pack('>%dI' % len(a), *a)

def get_chunks(size):
    p, s = 0, 0x20000

    while p + s < size:
        yield(p, s)
        p += s

        if s < 0x100000: s += 0x20000

    yield(p, size - p)

def aes_cbc_decrypt(data, key):
    aes_cipher = AES.new(key, AES.MODE_CBC, makebyte('\0' * 16))
    return aes_cipher.decrypt(data)

def decrypt_attr(attr, key):
    attr = codecs.latin_1_decode(aes_cbc_decrypt(attr, a32_to_str(key)))[0].rstrip('\0')
    return json.loads(attr[4:]) if attr[:6] == 'MEGA{"' else False

def _api_request(data):
    sequence_num = random.randint(0, 0xFFFFFFFF)
    params = {'id': sequence_num}
    sequence_num += 1

    if not isinstance(data, list): data = [data]
    json_resp = json.loads(requests.post('{0}://g.api.{1}/cs'.format('https', 'mega.co.nz'), params=params, data=json.dumps(data), timeout=160).text)
    if isinstance(json_resp, int): raise Exception(json_resp)

    return json_resp[0]

def base64_url_decode(data):
    data += '=='[(2 - len(data) * 3) % 4:]

    for search, replace in (('-', '+'), ('_', '/'), (',', '')):
        data = data.replace(search, replace)

    return base64.b64decode(data)

def str_to_a32(b):
    if isinstance(b, str): b = makebyte(b)
    if len(b) % 4: b += b'\0' * (4 - len(b) % 4)
    return struct.unpack('>%dI' % (len(b) / 4), b)

def base64_to_a32(s):
    return str_to_a32(base64_url_decode(s))

def mega_download_file(file_handle, file_key, dest_path=None):
    file_key = base64_to_a32(file_key)
    file_data = _api_request({'a': 'g', 'g': 1, 'p': file_handle})

    k = (file_key[0] ^ file_key[4], file_key[1] ^ file_key[5], file_key[2] ^ file_key[6], file_key[3] ^ file_key[7])
    iv = file_key[4:6] + (0, 0)

    if 'g' not in file_data: raise Exception(translations["file_not_access"])

    file_size = file_data['s']
    attribs = decrypt_attr(base64_url_decode(file_data['at']), k)
    input_file = requests.get(file_data['g'], stream=True).raw

    temp_output_file = tempfile.NamedTemporaryFile(mode='w+b', prefix='megapy_', delete=False)
    k_str = a32_to_str(k)
    aes = AES.new(k_str, AES.MODE_CTR, counter=Counter.new(128, initial_value=((iv[0] << 32) + iv[1]) << 64))

    mac_str = b'\0' * 16
    mac_encryptor = AES.new(k_str, AES.MODE_CBC, mac_str)
    iv_str = a32_to_str([iv[0], iv[1], iv[0], iv[1]])

    with tqdm.tqdm(total=file_size, ncols=100, unit="byte") as pbar:
        for _, chunk_size in get_chunks(file_size):
            chunk = aes.decrypt(input_file.read(chunk_size))
            temp_output_file.write(chunk)
            pbar.update(len(chunk))
            encryptor = AES.new(k_str, AES.MODE_CBC, iv_str)

            for i in range(0, len(chunk) - 16, 16):
                block = chunk[i:i + 16]
                encryptor.encrypt(block)

            i = (i + 16) if file_size > 16 else 0
            block = chunk[i:i + 16]
            if len(block) % 16: block += b'\0' * (16 - (len(block) % 16))

            mac_str = mac_encryptor.encrypt(encryptor.encrypt(block))

    file_mac = str_to_a32(mac_str)
    temp_output_file.close()

    if (file_mac[0] ^ file_mac[1], file_mac[2] ^ file_mac[3]) != file_key[6:8]: raise ValueError(translations["mac_not_match"])

    file_path = os.path.join(dest_path, attribs['n'])
    if os.path.exists(file_path): os.remove(file_path)

    shutil.move(temp_output_file.name, file_path)
    return file_path

def mega_download_url(url, dest_path=None):
    if '/file/' in url:
        url = url.replace(' ', '')
        file_id = re.findall(r'\W\w\w\w\w\w\w\w\w\W', url)[0][1:-1]
        path = f'{file_id}!{url[re.search(file_id, url).end() + 1:]}'.split('!')
    elif '!' in url: path = re.findall(r'/#!(.*)', url)[0].split('!')
    else: raise Exception(translations["missing_url"])

    return mega_download_file(path[0], path[1], dest_path)