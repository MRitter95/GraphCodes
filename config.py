import site
import os
import sys

site_directory = site.getusersitepackages()
package_directory = os.getcwd()

filepath = os.path.join(site_directory+"\kollarlab.pth")

print(site_directory)
print(package_directory)
print(filepath)

f = open(filepath,"w")
f.write(package_directory)