from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, FloatField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Length

class DataUploadForm(FlaskForm):
    """数据上传表单"""
    data_file = FileField('数据文件', validators=[
        FileRequired(message='请选择数据文件'),
        FileAllowed(['csv', 'xlsx', 'xls'], message='只支持CSV和Excel格式')
    ])
    submit = SubmitField('上传数据')

class ModelUploadForm(FlaskForm):
    """模型上传表单"""
    model_file = FileField('模型文件', validators=[
        FileRequired(message='请选择模型文件'),
        FileAllowed(['pkl'], message='只支持.pkl格式')
    ])
    model_name = StringField('模型名称', validators=[
        DataRequired(message='请输入模型名称'),
        Length(min=1, max=100, message='模型名称长度应在1-100字符之间')
    ])
    submit = SubmitField('上传模型')

class OptimizationForm(FlaskForm):
    """参数优化表单"""
    model_select = SelectField('选择模型', choices=[
        ('rbfnn', 'RBFNN预测模型 v3.2'),
        ('lasso', 'Lasso预测模型 v2.5'),
        ('efficiency', '效率优化模型 v1.7')
    ], validators=[DataRequired()])
    
    objective = SelectField('优化目标', choices=[
        ('quality', '最大化质量评分'),
        ('quantity', '最大化产量'),
        ('cost', '最小化成本'),
        ('balance', '平衡质量与产量')
    ], validators=[DataRequired()])
    
    algorithm = SelectField('算法选择', choices=[
        ('PSO', '粒子群优化 (PSO)'),
        ('GA', '遗传算法 (GA)'),
        ('SA', '模拟退火 (SA)'),
        ('GD', '梯度下降 (GD)')
    ], validators=[DataRequired()])
    
    # 参数范围
    rate_min = FloatField('大中片率最小值', validators=[
        DataRequired(), NumberRange(min=70, max=90)
    ], default=77)
    
    rate_max = FloatField('大中片率最大值', validators=[
        DataRequired(), NumberRange(min=70, max=90)
    ], default=79)
    
    leaf_stem_min = FloatField('叶含梗率最小值', validators=[
        DataRequired(), NumberRange(min=0.5, max=2.0)
    ], default=1.1)
    
    leaf_stem_max = FloatField('叶含梗率最大值', validators=[
        DataRequired(), NumberRange(min=0.5, max=2.0)
    ], default=1.5)
    
    stem_leaf_min = FloatField('梗含叶率最小值', validators=[
        DataRequired(), NumberRange(min=0.3, max=1.0)
    ], default=0.5)
    
    stem_leaf_max = FloatField('梗含叶率最大值', validators=[
        DataRequired(), NumberRange(min=0.3, max=1.0)
    ], default=0.7)
    
    submit = SubmitField('开始优化')

class PredictionForm(FlaskForm):
    """模型预测表单"""
    flow_rate = FloatField('铺叶工段数据记录_流量（kg/h)', validators=[
        DataRequired(), NumberRange(min=10000, max=20000)
    ], default=15450)
    
    moisture_after = FloatField('二润工段数据记录_润后水份（%）', validators=[
        DataRequired(), NumberRange(min=10, max=30)
    ], default=18.91)
    
    temp_after = FloatField('二润工段数据记录_润后温度（℃）', validators=[
        DataRequired(), NumberRange(min=40, max=70)
    ], default=54.5)
    
    moisture_before = FloatField('打叶工段数据记录_打前水份（%）', validators=[
        DataRequired(), NumberRange(min=10, max=25)
    ], default=15.8)
    
    submit = SubmitField('运行预测')