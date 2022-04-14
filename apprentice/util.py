import inspect

class Util(object):

    @staticmethod
    def inherits_from(child, parent_name):
        if inspect.isclass(child.__class__):
            if parent_name in [c.__name__ for c in inspect.getmro(child.__class__)[1:]]:
                return True
        return False

