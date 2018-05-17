import rrdtool
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from threading import Lock
from pathlib import PurePosixPath


class SensorData():
    def __init__(self, db):
        self._db = db
        self._data = rrdtool.lastupdate(db)
        self._images = {}

    @property
    def date(self):
        return self._data['date']

    def image(self, path):
        try:
            image = self._images[path]
        except KeyError:
            # generate it
            p = PurePosixPath(path)
            try:
                element, duration = p.stem.split('_', 1)
            except ValueError:
                raise KeyError(path)
            start = {
                'recent':  '1d',
                'history': '1M',
            }[duration]
            color = {
                'temperature': '#FF0000',
                'humidity':    '#0000FF',
                'pressure':    '#00FF00',
            }[element]
            self._images[path] = image = rrdtool.graphv(
                '-',
                '--imgformat', 'SVG',
                '--border', '0',
                '--color', 'BACK#00000000', # transparent
                '--start', 'now-' + start,
                '--end', 'now',
                'DEF:v={db}:{element}:AVERAGE'.format(db=self._db, element=element),
                'LINE2:v{color}'.format(color=color)
            )['image']
        return image

    def __format__(self, format_spec):
        element, units = format_spec.split(':')
        template = """
<div class="sensor">
    <h2>{title}</h2>
    <span class="reading">{current:.1f}{units}</span>
    <img class="recent" src="{element}_recent.svg" />
    <img class="history" src="{element}_history.svg" />
</div>
"""
        return template.format(
            element=element,
            title=element.title(),
            units=units,
            current=self._data['ds'][element])


class RequestHandler(BaseHTTPRequestHandler):
    database = 'environ.rrd'
    data = None
    index_template = """
<html>
    <head>
        <title>Sense HAT Environment Sensors</title>
        <link href="https://fonts.googleapis.com/css?family=Raleway" rel="stylesheet">
        <style>
body {{
    font-family: "Raleway", sans-serif;
    max-width: 700px;
    margin: 1em auto;
}}

h1 {{ text-align: center; }}

div {{
    padding: 8px;
    margin: 1em 0;
    border-radius: 8px;
}}

div#timestamp {{
    font-size: 16pt;
    background-color: #bbf;
    text-align: center;
}}

div.sensor {{ background-color: #ddd; }}

div.sensor h2 {{
    font-size: 20pt;
    margin-top: 0;
    padding-top: 0;
    float: left;
}}

span.reading {{
    font-size: 20pt;
    float: right;
    background-color: #ccc;
    border-radius: 8px;
    box-shadow: inset 0 0 4px black;
    padding: 4px 8px;
}}
        </style>
    </head>
    <body>
        <h1>Sense HAT Environment Sensors</h1>
        <div id="timestamp">{data.date:%A, %d %b %Y %H:%M:%S}</div>
        {data:temperature:°C}
        {data:humidity:%RH}
        {data:pressure:mbar}
        <script>
        setTimeout(() => location.reload(true), 10000);
        </script>
    </body>
</html>
"""

    def get_sensor_data(self):
        # Keep a copy of the latest SensorData around for efficiency
        old_data = RequestHandler.data
        new_data = SensorData(RequestHandler.database)
        if old_data is None or new_data.date > old_data.date:
            RequestHandler.data = new_data
        return RequestHandler.data

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            data = self.get_sensor_data()
            content = RequestHandler.index_template.format(
                data=data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(content))
            self.send_header('Last-Modified', self.date_time_string(
                data.date.timestamp()))
            self.end_headers()
            self.wfile.write(content)
        elif self.path.endswith('.svg'):
            data = self.get_sensor_data()
            try:
                content = data.image(self.path)
            except KeyError:
                self.send_error(404)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'image/svg+xml')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
        else:
            self.send_error(404)

def main():
    httpd = HTTPServer(('', 8000), RequestHandler)
    httpd.serve_forever()

if __name__ == '__main__':
    main()
