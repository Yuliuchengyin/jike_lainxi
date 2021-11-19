# 一  数组


![在这里插入图片描述](https://img-blog.csdnimg.cn/dc93515af2ec4686b430dd1aa33c436f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

@[TOC](目录)
#### <font color=red face="微软雅黑">来源</font>

Datewhle31期__LeetCode 刷题 :
* 航路开辟者：杨世超
* 领航员：刘军
* 航海士：杨世超、李彦鹏、叶志雄、赵子一

* 开源内容
<https://github.com/itcharge/LeetCode-Py>
* 开源电子书
<https://algo.itcharge.cn>




## 1.1 基础理论
>**数组（Array**）：一种线性表数据结构。它使用一组连续的内存空间，来存储一组具有相同类型的数据。


* <font color=red face="微软雅黑">**数组** 就是==相同类型==的数据元素构成的==有序集合==.</font>

* **数组**常与**链表**比较: 
![](https://img-blog.csdnimg.cn/2aeaea2be77245fe9acd980814a15e77.png)
可见数组的特点时可根据**下标及寻址公式**,  ==进行随机访问且效率高==

>寻址公式如下：下标 i 对应的数据元素地址 = 数据首地址 + i * 单个数据元素所占内存大小
------

* 一维及多维数组图例:  

![在这里插入图片描述](https://img-blog.csdnimg.cn/eadc820c4bd245e0b083594456f597b7.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



* 以二维数组为例，数组的形式如下图所示。


![在这里插入图片描述](https://img-blog.csdnimg.cn/8dc9c725ff2b46c2b8abda31dba1d725.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)





----


* ==数组在C, java, python的实现==

  1.  **C**
数组都是由**连续的内存位置**组成。最低的地址对应第一个元素，最高的地址对应最后一个元素。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/13894279698b40c6b29cc3ab6245b829.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
<font color=red face="微软雅黑">不管是基本类型数据，还是结构体、对象，在数组中都是连续存储的</font>
如:
```cpp
int arr[3][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/85d940ea98c14ad0b90c985ec3570c6c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
C中在内存中是连续存储的,  如上图，一个二维数组就可以看成一个一维数组，只是里面存放的元素为一维数组。==所以C中的数组是呈线性结构==		。

  2. **Java**
  Java 中的数组也是存储相同类型数据的，但所使用的内存空间却**不一定是连续（多维数组中）**。且如果是多维数组，其嵌套数组的长度也可以不同。例如：
  ```java
  int[][] arr = new int[3][]{ {1,2,3}, {4,5}, {6,7,8,9}};
  ```
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/243b136fd4664105833165aab98c2152.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

如上图, 在Java中，数组都是引用实体变量，==呈树形结构，每一个叶子节点之间毫无关系，只有引用关系，每一个引用变量只引用一个实体==。
  3. Python
  	原生 Python 中其实没有数组的概念，而是使用了类似 Java 中的 ArrayList 容器类数据结构，叫做列表。通常我们把列表来作为 Python 中的数组使用。Python 中列表==存储的数据类型可以不一致，数组长度也可以不一致==。例如：
```python
  	arr = ['python', 'java', ['asp', 'php'], 'c']
 ```




* ==数组的操作---增查改删==

1. 访问及查找
   *  访问数组中第 `i` 个元素：只需要检查` i `的范围是否在合法的范围区间, **时间复杂度为O(1)**
   * 查找数组中元素值为 val 的位置：在数组无序的情况下，只能通过将 val 与数组中的数据元素逐一对比的方式进行检索，也称为**线性查找**------------**时间复杂度为O(n)** 。
2.  插入
    * ==尾部插入==:如果数组尾部容量不满，**则直接把 val 放在数组尾部的空闲位置**，并更新数组的元素计数值。如果数组**容量满了，则插入失败**。
    不过，`Python` 中的 `list` 做了其他处理，当数组容量满了，则会开辟新的空间进行插入。在尾部插入元素的操作不依赖数组个数，**其时间复杂度为O(1)** 。
    python用append即可/:
```python
arr = [0, 5, 2, 3, 7, 1, 6]
val = 4
arr.append(val)
print(arr)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/12e5f5e432424d64bd5f82f3fcf42586.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)              
      2.  ==第 `i` 个位置上插入==:  先检查插入下标 i 是否合法，即 `0 <= i <= len(nums)`。确定合法位置后，通常情况下第 i 个位置上已经有数据了（除非` i == len(nums)` ），要把第 i 个位置到第` len(nums) - 1` 位置上的元素依次向后移动，然后再在第 i 个元素位置插入 val 值，并更新数组的元素计数值。因为移动元素的操作次数跟元素个数有关，**最坏和平均时间复杂度都是O(n)**。

Python 中的 list 直接封装了中间插入操作，直接调用 insert 方法即可。

```python
arr = [0, 5, 2, 3, 7, 1, 6]
i, val = 2, 4	
arr.insert(i, val)
print(arr)
```

  
3. ==改变==:
   * 将数组中第 i 个元素值改为 val：**改变元素操作跟访问元素操作类似**。需要先检查 i 的范围是否在合法的范围区间，即 0 <= i <= len(nums) - 1。然后将第 i 个元素值赋值为 val。访问操作不依赖于数组中元素个数，**因此时间复杂度为O(1)** 。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/ff088a8b7c1b4f4db2bc1d416571c963.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
 
4. ==删除==
    *  删除数组尾部-----------------将元素计数值减一即可, 时间复杂度为 O(1)。------python中使用`pop`
    * 删除数组第 i 个位置上的元素---------先检查下标 i 是否合法，即 `o <= i <= len(nums) - 1`。如果下标合法，则将第 i + 1 个位置到第 len(nums) - 1 位置上的元素依次向**左移动**。删除后修改数组的元素计数值。删除中间位置元素的操作同样涉及移动元素，而移动元素的操作次数跟元素个数有关，因此删除中间元素的最坏和平均时间复杂度都是O(n)-------------同样使用`pop`

![在这里插入图片描述](https://img-blog.csdnimg.cn/e96ad183d3de4f41ba4afc5a8ed927fb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
  3.  基于条件删除元素 :------------------**通过循环检查元素，查找到元素后将其删除**。删除多个元素操作中涉及到的**多次移动元素操作**，可以通过算法改进，将多趟移动元素操作转变为一趟移动元素，从而将时间复杂度降低为O(n) 。一般而言，这类删除操作都是线性时间操作，时间复杂度为O(n) 。---------------`remove`


```python
arr = [0, 5, 2, 3, 7, 1, 6]
i = 3
arr.remove(5)
print(arr)
```



## 1.2 数组基础题目
### 1.2.1 189旋转数组

<https://leetcode-cn.com/problems/rotate-array/>



* 法一--------自己第一想到----------O(n)
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k %= len(nums)
        for i in range(k): 
            nums.insert(0,nums[-1])
            nums.pop()
```
* 法二-------三次颠倒-----------------也是参考答案

![在这里插入图片描述](https://img-blog.csdnimg.cn/cc5a57173aff4985bf6d63980542fb14.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k %= len(nums)
        nums[:] = nums[::-1]       
        nums[:k] = nums[:k][::-1]
        nums[k:] = nums[k:][::-1]
```





* 法三-------拼接------------与二类似

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k %= len(nums)
        nums[:] = nums[-k:] + nums[:-k] 
```

	




![在这里插入图片描述](https://img-blog.csdnimg.cn/3242dd3294984995962af0ccaf68d1a3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

### 1.2.2 66加一

<https://leetcode-cn.com/problems/plus-one/>

* 法一--------------直接想到的----------转换成int+1  再转回列表

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        dig = int(''.join(str(i) for i in digits)) + 1
        return [int(i) for i in list(str(dig))]
```



* 法二-------------参考答案-----------首位补0 + 尾巴加1 循环判断尾巴是否为9

==想到尾巴加一思路-------但自己不知如何处理1999999这种情况---------前面补0太秒==

思路: 这道题把整个数组看成了一个整数，然后个位数 +1。问题的实质是利用数组模拟加法运算。

如果个位数不为 9 的话，直接把个位数 +1 就好。如果个位数为 9 的话，还要考虑进位。

具体步骤：

1. 数组前补 0 位。
2. 将个位数字进行 +1 计算。
3. 遍历数组
        1.如果该位数字 大于等于 10，则向下一位进 1，继续下一位判断进位。
        2.如果该位数字 小于 10，则跳出循环。
```python
def plusOne(self, digits: List[int]) -> List[int]:
    digits = [0] + digits
    digits[len(digits)-1] += 1
    for i in range(len(digits)-1,0,-1):
        if digits[i] != 10:
            break
        else:
            digits[i] = 0
            digits[i-1] += 1
        
    if digits[0] == 0:
        return digits[1:] 
    else:
        return digits
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/06e53aadac99436aa56453ca86314d6c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


### 1.2.3  724 . 寻找数组的中心下标

<https://leetcode-cn.com/problems/find-pivot-index/>




法1 -------------  一开始想到的 利用for循环 条件为 sum(:i) == sum(i+1:)


```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            if sum(nums[:i]) == sum(nums[i+1:]):
                return i
        return -1
```


* 结果时间爆炸-------------------  6684ms!!!!!!
 
==由于没刷题经验....经实验   →    for里面加sum() \=  完犊子==


* 看参考答案总结利用 →  两边和(sum左或右) + 目前值(num[i]) =  列表总和(tatol)  →  即可


* 法二 

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        sum_1 = 0
        total = sum(nums)     #注意:  求和位于for之外
        for i in range(len(nums)):
            if sum_1 * 2 + nums[i] == total:
                return i
            sum_1 += nums[i]
        return -1
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2ddfb22acae6438ab695701bd3e672b9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



### 1.2.4  0485. 最大连续 1 的个数
<https://leetcode-cn.com/problems/max-consecutive-ones/>


* 一个暂存计数(sum_1),   另一用来返回最大计数(count)

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        count = 0
        sum_1 = 0
        for i in nums:
            if i == 1:
                sum_1 += 1
                count = max(sum_1, count)
            else:
                sum_1 = 0   #碰到0重新计数
        return count
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/5421a517d6704cd8a3fa8131fa9f8e31.png)


### 1.2.5  238 . 除自身以外数组的乘积
<https://leetcode-cn.com/problems/product-of-array-except-self/>



* 思路:   num[i]左边乘积  与右边乘积     相乘 →     for循环出两边乘积

![在这里插入图片描述](https://img-blog.csdnimg.cn/4fa8cabbf96841b28cb563131d084cda.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        right = 1
        left = 1
        output = [1 for _ in range(len(nums))]
        for i in range(len(nums)):
            output[i] *= left
            left *= nums[i]

        for j in range(len(nums)-1, -1 , -1):
            output[j] *= right
            right *= nums[j]


        return output
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/9f523590cb834e8a8c3557434557ace7.png)
### 1.2.6 面试题 01.07 . 旋转矩阵
<https://leetcode-cn.com/problems/rotate-matrix-lcci/>

* ==方法一==  →→→→→首先想到用numpy  →  先拆分再组合(**实际不符题意---占用额外的空间**且速度超级慢)

```python
import numpy as np
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        
        m = np.array(matrix)     #转换成array数组
        n = m.shape[0]
        m = np.split(m,n,axis = 0)    #将原数组按行拆分
        m = np.stack(m[::-1],axis = 2)   # 将拆分结构翻转并组合
        matrix[:] = m.tolist()[0]
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/cb6a764f78f04a999faf5e7dbb41c1cb.png)
* 于是随便复习一下`numpy`的一些用法:
`stack`:   Numpy中stack()，hstack()，vstack()函数详解<https://blog.csdn.net/csdn15698845876/article/details/73380803?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.essearch_pc_relevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.essearch_pc_relevant>
* `split`  <https://numpy.org/doc/stable/reference/generated/numpy.split.html>


* [:]表示浅拷贝   <https://zhuanlan.zhihu.com/p/57893374>
----



* ==法二== →→→→→ 同样numpy  →→ 参考评论`numpy.rot90`的用法:
`rot90`  :<https://numpy.org/doc/stable/reference/generated/numpy.rot90.html>


> 一行即解决---------但时间和内存和法一差不多
```python
matrix[:] = np.rot90(np.array(matrix), -1).tolist()
```


*  ==法三==  →→→→→ 上下翻 , 再对角翻  →参考答案→ 较符合题意且符合算法逻辑



```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n//2):
            for j in range(n):
                matrix[i][j],matrix[n-1-i][j] = matrix[n-1-i][j],matrix[i][j]
        for i in range(n):
            for j in range(i):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
```




![在这里插入图片描述](https://img-blog.csdnimg.cn/23599c124f694c9a91e7d679dc3bec5d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



### 1.2.7 面试题 01.08 . 零矩阵
<https://leetcode-cn.com/problems/zero-matrix-lcci/>


* ==法一==--------------  先想到记录0的row和col  再分别将行列变为0


```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        count_x = set()
        count_y = set()
        n_x = len(matrix)
        n_y = len(matrix[0])

        for i in range(n_x):
            for j in range(n_y):
                if matrix[i][j] == 0:
                    count_x.add(i)
                    count_y.add(j)

        for i in count_x:
            matrix[i] = [0 for _ in range(n_y)]


        for i in range(n_x):
            for j in count_y:
                matrix[i][j] = 0
```



* ==法二== --------------------参考答案-----------利用原本空间即第一行第一列的bool来记录0-----------妙妙妙!



```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        rows = len(matrix)
        cols = len(matrix[0])

        row_0 = False
        col_0 = False    #第一行及第一列记录所对应 行列 是否有0元素  ---不另外占空间 

        for i in range(rows):
            if matrix[i][0] == 0:
                col_0 = True     #首列存在0元素,  首列 清零开关打开
                break

        for j in range(cols):
            if matrix[0][j] == 0:
                row_0 = True     #首行存在0元素,  首行 清零开关打开
                break


        for i in range(1, rows):
            for j in range(1, cols):   
                if matrix[i][j] == 0:
                    matrix[0][j] = matrix[i][0] = 0  #将元素为0对应的首行, 首列元素改为0

        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[0][j] == 0 or matrix[i][0] == 0:  #如果首行首列为0, 对应行列全改为0
                    matrix[i][j] = 0

        
        if col_0:
            for i in range(rows):        #改首列全为0
                matrix[i][0] = 0 


        if row_0:
            for j in range(cols):     
                matrix[0][j] = 0    
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/4082755a57b94cff81c132de7ca8bbec.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



----

### 1.2.8 498 . 对角线遍历
<https://leetcode-cn.com/problems/diagonal-traverse/>

* 好难----------一直没找着规律---  下标相加之和递增 →  且分 奇偶: 下标和偶向上;  下标和奇数向下:


* 找规律：

   * 当行号 + 列号为偶数时，遍历方向为从左下到右上。可以记为右上方向（-1, +1），即行号 -1，列号 +1。
   * 当行号 + 列号为奇数时，遍历方向为从右上到左下。可以记为左下方向（+1, -1），即行号 +1，列号 -1。

边界情况：

* 向右上方向移动时：
   * 如果在最后一列，则向下方移动，即 x += 1。
   * 如果在第一行，则向右方移动，即 y += 1。
   * 其余情况想右上方向移动，即 x -= 1、y += 1。
* 向左下方向移动时：
   * 如果在最后一行，则向右方移动，即 y += 1。
   * 如果在第一列，则向下方移动，即 x += 1。
   * 其余情况向左下方向移动，即 x += 1、y -= 1。


```python
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        rows = len(mat)
        cols = len(mat[0])
        count = rows * cols
        x, y = 0, 0
        ans = []

        for i in range(count):
            ans.append(mat[x][y])

            if (x + y) % 2 == 0:
                # 最后一列
                if y == cols - 1:
                    x += 1
                # 第一行
                elif x == 0:
                    y += 1
                # 右上方向
                else:
                    x -= 1
                    y += 1
            else:
                # 最后一行
                if x == rows - 1:
                    y += 1
                # 第一列
                elif y == 0:
                    x += 1
                # 左下方向
                else:
                    x += 1
                    y -= 1
        return ans
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/c0906267af8242f38437ecd25a145fb9.png)


### 1.2.9 498 . 对角线遍历
* 规律: `matrixnew[col][n−row−1] = matrix[row][col]`

* 方法与旋转矩阵一样: 先上下翻, 再对角翻



### 1.2.10 118 . 杨辉三角
<https://leetcode-cn.com/problems/pascals-triangle/>


* 简单---

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        n = []
        for i in range(numRows):
            tmp = [1 for _ in range(i + 1)]  # 将每一行填充为全为一的list
            n.append(tmp)
            for j in range(1,i):
                n[i][j] = n[i-1][j] + n[i-1][j-1]  # 当前元素等于上两个之和
        return n
```



### 1.2.11 119 . 杨辉三角二
<https://leetcode-cn.com/problems/pascals-triangle-ii/>

* 法一:  numRows+1 , 返回以上代码 n[-1]即可

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        n = []
        for i in range(rowIndex + 1):
            tmp = [1 for _ in range(i + 1)]  # 将每一行填充为全为一的list
            n.append(tmp)
            for j in range(1,i):
                n[i][j] = n[i-1][j] + n[i-1][j-1]  # 当前元素等于上两个之和
        return n[-1]
```
	
### 1.2.12 73 . 矩阵置零
<https://leetcode-cn.com/problems/set-matrix-zeroes/>






## 参考资料


1. <https://zhuanlan.zhihu.com/p/105962783>-----什么是数组？
2. <https://www.runoob.com/cprogramming/c-arrays.html> C数组
3. <https://blog.csdn.net/qq_42913794/article/details/89077825>Java和C的数组区别
4. <https://zhuanlan.zhihu.com/p/57893374>浅拷贝和深拷贝

