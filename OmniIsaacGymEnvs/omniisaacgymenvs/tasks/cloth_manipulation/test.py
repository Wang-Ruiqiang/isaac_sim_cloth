import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 

dataX = [1,2,3,4]
dataY = [1,2,3,1]
plt.plot(dataX,dataY)#plot还有很多参数，可以查API修改，如颜色，虚线等
plt.title("绘制直线")
plt.xlabel("x轴")
plt.ylabel("y轴")
plt.show()