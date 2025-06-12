import panel as pn
import param
from panel.viewable import Viewer


#class Settings(param.Parameterized):
class Settings(Viewer):
    integer = param.Integer(default=1, bounds=(0, 10))
    string = param.String(default='A string')

    dont_sync = param.String(default='A string')

    def __init__(self,**params):
        super().__init__(**params);
    
    def __panel__(self):
        return pn.Column(
            pn.Param(self),
            'HI There {:s}'.format(self.string)
        )


settings = Settings()

pn.state.location.sync(settings, {'integer': 'int', 'string': 'str'})

#pn.Param(settings).servable()
settings.servable()
