import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import json

from raytracer import render, optimize, pleasant_defaults
from scipy.misc import toimage
import StringIO
import numpy

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print 'new connection'
        self.send_render(render(pleasant_defaults), pleasant_defaults)

    def optimize(self, params, x, y, lr):
        for step in optimize(params, x, y, lr, self.send_status):
            self.send_render(step[0], step[1])

    def send_render(self, image, params):
        message = {
            'type': 'render',
            'image': self.encode(image),
            'params': params
        }

        self.write_message(json.dumps(message, cls=NumpyAwareJSONEncoder))

    def send_status(self, status):
        message = {
            'type': 'status',
            'status': status
        }

        self.write_message(json.dumps(message))

    def encode(self, image):
        output = StringIO.StringIO()
        im = toimage(image)
        im.save(output, 'png')
        return output.getvalue().encode("base64")

    def on_message(self, message):
        print 'message received %s' % message
        args = json.loads(message)
        self.optimize(args['params'], args['x'], args['y'], args['lr'])

    def on_close(self):
        print 'connection closed'


application = tornado.web.Application([
    (r'/ws', WSHandler),
])

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
