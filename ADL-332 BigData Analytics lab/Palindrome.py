s = input()
if s == s[::-1]:
    print("Palindrome")
else:
    print("Not Palindrome")



/// Another way 
s=input()
rev=""
for i in s:
  rev=i+rev
if rev==s:
  print("IS palindrome")
else:
  print("Not a palindrome")
