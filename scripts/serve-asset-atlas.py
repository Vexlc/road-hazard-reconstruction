"""Serve the local RoadGS atlas, registered assets and supporting evidence on loopback."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
START = '/tools/asset_atlas/index.html'
PUBLIC = [ROOT/d for d in ('tools/asset_atlas','data/releases','docs','runs','outputs','validation','gen2hole/docs')]


class AtlasHandler(SimpleHTTPRequestHandler):
    def translate_path(self,url):
        path=Path(super().translate_path(url)).resolve()
        if not any(path.is_relative_to(root) for root in PUBLIC):
            return str(ROOT/'tools/asset_atlas'/'unavailable-resource')
        return str(path)

    def list_directory(self,path):
        self.send_error(404)
        return None

    def send_head(self):
        if urlsplit(self.path).path=='/':
            self.send_response(302)
            self.send_header('Location',START)
            self.end_headers()
            return None
        return super().send_head()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    handler=partial(AtlasHandler,directory=ROOT)
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
        print(f'http://127.0.0.1:{server.server_port}{START}',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
