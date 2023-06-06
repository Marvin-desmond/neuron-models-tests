import requests 
from pathlib import Path 
import urllib
import subprocess
import torch 

def curl_download(url, filename, *, silent: bool = False) -> bool:
    """
    Download a file from a url to a filename using curl.
    """
    silent_option = 'sS' if silent else ''  # silent
    proc = subprocess.run([
        'curl',
        '-#',
        f'-{silent_option}L',
        url,
        '--output',
        filename,
        '--retry',
        '9',
        '-C',
        '-',])
    return proc.returncode == 0

def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=True)
        if file.exists():
            print(f"{file} : {file.stat().st_size}")
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e: 
        if file.exists():
            file.unlink()  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            return False
        else:
            print(f'ERROR: {assert_msg}\n{error_msg}')
            return True

def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    downloaded = False
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v7.0', etc.
    def github_assets(repository, version='latest'):
        # Return GitHub repo tag (i.e. 'v7.0') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v7.0
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    file = Path(str(f"./sample-ultralytics/{file}").strip())

    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name 

        # GitHub assets
        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]  # default

        try:
            # Tries to get assets from the default release name in root function
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                # Tries to download assets from the latest release (if release is null, latest becomes default)
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    # Tries to get tags from the repo, then take the last one (the latest one)
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception as e:
                    # If all else fails, revert to default release name in root function
                    tag = release

        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)
            downloaded = safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag}')
        else:
            print(f"{name} not found in {repo}")
    else:
        downloaded = True
    return str(file), downloaded


files = [
    'yolov5l.pt',
    'yolov5l6.pt',
    'yolov5m-VOC.pt',
    'yolov5m.pt',
    'yolov5m6.pt',
    'yolov5n-7-k5.pt',
    'yolov5n-7.pt',
    'yolov5n.pt',
    'yolov5n6.pt',
    'yolov5s.pt',
    'yolov5s6.pt',
    'YOLOv5x-7-k5.pt',
    'YOLOv5x-7.pt',
    'yolov5x.pt',
    'yolov5x6.pt',
]