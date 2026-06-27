# Pencil Price Prediction Using Linear Regression----
X=[10,12,15,17,5]
Y=[6,8,20,6,25]
  

n=len(X)
#Calculating Average
X_bar=sum(X)/n
Y_bar=sum(Y)/n

#Taking Components of m
num=0
den=0

for i in range (n):
    num+=(X[i]-X_bar)*(Y[i]-Y_bar)
    den+=(X[i]-X_bar)**2

#Calculating m & c 
m = num / den
C = Y_bar-m*X_bar
print("Slope(m)=",round(m,4))
print("Intercept(c)=",round(m,4))

#Taking input from user:-
length =float(input("Enter the length of pencil:"))
price=m*length+C
print("Predicted Price = ", round(price,2))
