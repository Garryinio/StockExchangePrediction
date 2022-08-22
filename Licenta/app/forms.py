from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Length


class DataForm(FlaskForm):
    dataSet = StringField('Stock', validators=[Length(min=1,max=5)])
    submit = SubmitField('Alege data set-ul')

#DataRequired(),