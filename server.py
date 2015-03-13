import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import json

from scenemaker import simple_scene
from optimize import GDOptimizer
from scipy.misc import toimage
import StringIO
import numpy
import theano

scene = simple_scene()
opt = GDOptimizer(scene)

print "Rendering initial scene"
variables, values, image = scene.build()
current = theano.function(variables, image, on_unused_input='ignore')(*values)
print "Done"


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class WSHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print 'new connection'
        self.send_render(current)

    def optimize(self, x, y, lr):
        for step in opt.optimize(image[y, x].sum(), lr, self.send_status):
            current = step
            self.send_render(step)

    def send_render(self, image):
        message = {
            'type': 'render',
            'image': self.encode(image)
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
        self.optimize(args['x'], args['y'], args['lr'])

    def on_close(self):
        print 'connection closed'


application = tornado.web.Application([
    (r'/ws', WSHandler),
])

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
