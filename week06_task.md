# 五、二分, 排序

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----


@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 二分法
### 1.1 基本思想
>二分查找又称**折半查找**，它是一种效率较高的查找方法。

>>**二分查找要求**
>* 目标函数具有单调性(单调递增或者递减) 
>* 存在上下界(bounded) 
>* 能够通过索引访问(indexaccessible)


### 1.2 **直接找法**

==直接找法==:  在循环体中找到元素就直接返回结果----若`nums[mid]`值大于或小于目标值, 分别取区间`[left, mid - 1 ]` 或`[mid + 1 , right ]`>>>>>>><font color=red face="微软雅黑" size =2.5>适用用于简单的,非重复的元素查找值问题</font>


* python实现
  * 以[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)为例;

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left,right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1 
            else:
                left = mid + 1
        return -1
```
---
==两个细节==
1.  **mid的取值问题**
常见的 mid 取值就是 `mid = (left + right) // 2` 或者 `mid = left + (right - left) // 2` 。前者是最常见写法，后者是为了防止整型溢出。式子中 `// 2` 就代表的含义是中间数「**向下取整**」------- 取左。
`mid = (left + right + 1) // 2`，或者 `mid = left + (right - left + 1) // 2`可以做到取右, 但一般来说，取中间位置元素在平均意义下所达到的效果最好。同时这样写最简单。而对于 mid 值是向下取整还是向上取整，==大多数时候是选择不加 1==.
2. **边界条件问题**----------`left <= right` 和 `left < right`
   *  `left <= right`，且查找的元素不存在，则 `while` 判断语句出界条件是 `left == right + 1`，写成区间形式就是 `[right + 1, right]`，此时待查找区间为空，待查找区间中没有元素存在，所以==此时终止循环可以直接`return -1 `是正确的==。
   * 如果判断语句为`left < right`，且查找的元素不存在，则 `while` 判断语句出界条件是 `left == right`，写成区间形式就是 `[right, right]`。此时区间不为空，待查找区间还有一个元素存在，并不能确定查找的元素不在这个区间中，此时终止循环`return -1 `是错误的。 >>>>>>  ==改为  `return left if nums[left] == target else -1`,  且不用判断返回`left` 还是 `right`==.
即:

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left,right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1 
            else:
                left = mid + 1
        return left if nums[left] == target else -1
```


### 1.3 排除法(lower_bound and upper_ bound)


1. 查找 `lower_bound` (第一个`> =target` 的数)

区间分为`[mid + 1, right]`和` [left, mid]`;  而 `mid = left + (right - left) // 2`:

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        # 在区间 [left, right] 内查找 target
        while left < right:
            # 取区间中间节点
            mid = left + (right - left) // 2
            # nums[mid] 小于目标值，排除掉不可能区间 [left, mid]，在 [mid + 1, right] 中继续搜索
            if nums[mid] < target:
                left = mid + 1 
            # nums[mid] 大于等于目标值，目标元素可能在 [left, mid] 中，在 [left, mid] 中继续搜索
            else:
                right = mid    # nums[mid] >= target
        # 判断区间剩余元素是否为目标元素，不是则返回 -1
        return right if nums[right] == target else -1
```







2. 查找`upper_bound`  最后一个`<=target`的数

区间分为` [left, mid - 1]`和 `[mid, right]`;  而 `mid = left + (right - left + 1) // 2`:

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        # 在区间 [left, right] 内查找 target
        while left < right:
            # 取区间中间节点
            mid = left + (right - left + 1) // 2
            # nums[mid] 大于目标值，排除掉不可能区间 [mid, right]，在 [left, mid - 1] 中继续搜索
            if nums[mid] > target:
                right = mid - 1 
            # nums[mid] 小于等于目标值，目标元素可能在 [mid, right] 中，在 [mid, right] 中继续搜索
            else:                      # nums[mid] <= target
                left = mid
        # 判断区间剩余元素是否为目标元素，不是则返回 -1
        return right if nums[right] == target else -1
   ```



   * 区分被分为两部分： `[left, mid - 1] 与 [mid, right]` 时，`mid` 取值要向上取整。即 `mid = left + (right - left + 1) // 2`。因为如果当区间中只剩下两个元素时（此时 `right = left + 1`），一旦进入`left = mid` 分支，区间就不会再缩小了，下一次循环的查找区间还是`[left, right]`，就陷入了死循环。


### 1.4 相关题目
1  [153 . 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)



* ==思路== : 旋转后数组分成两段有序设为0 和 1,  >>> 相当于寻找出`lower_bound`


```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left)//2
            if nums[mid] < nums[right]:
                right = mid 
            else:
                left = mid + 1
        return nums[right] 
```



2.  [34 . 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)


* ==思路==: 开始位置为第一个大于`target`,  结束位置为最后一个<=`target`


```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        ans = []
        left, right = 0, len(nums)  
        while left < right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
        ans.append(right)

        left, right = -1, len(nums) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if nums[mid] <= target:
                left = mid
            else:
                right = mid - 1
        ans.append(right)
        if ans[0] > ans[1]:
            return [-1, -1]
        else:
            return ans
```




3.   [69 . Sqrt(x)](https://leetcode-cn.com/problems/sqrtx/)


* upper_bound类型

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left, right = 0, x
        while left < right:
            mid = (left + right + 1) // 2
            if mid * mid <= x:
                left = mid
            else:
                right = mid - 1
        return right
```





4. [162 . 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

* 三分法用于求单峰函数的极大值(或单谷函数的极小值) 
三分法也可用于求函数的局部极大/极小值 
要求：函数是分段严格单调递増/递减的(不能出现一段平的情况) 


* 以求单峰函数f的极大值为例，可在定义域［L, r］上取任意两点Imid, rmid
 •若f(lmid) <= f(rmid),则函数必然在Imid处单调递増,极值在［Imid, r］ 上 
 •若f(lmid) >f(rmid),则函数必然在rmid处单调递减，极值在［L rmid］上 


* Imid, rmid可取三等分点 
也可取Imid为二等分点，rmid为Lmid稍加一点偏移量 

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            lmid = (left + right) // 2
            rmid = lmid + 1
            if nums[lmid] < nums[rmid]:left = lmid + 1
            else:right = rmid - 1
        return right
```




5. [410 . 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/)


* ==思路==: 转化为判定问题  >>>> 给一个数值T, "m个子数组各自和的最大值＜=T〃是否合法

```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:

        def validate(nums, m, size):
            box, count = 0, 1
            for num in nums:
                if num + box <= size:
                    box += num
                else:
                    box = num
                    count += 1
            return count <= m

        left, right = 0, 0
        for num in nums:
            left = max(left, num)
            right += num

        while left <right:
            mid = (right + left) // 2
            if validate(nums, m, mid):
                right = mid
            else:
                left = mid + 1
        return right
```




6. [1482 . 制作 m 束花所需的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-make-m-bouquets/)


```python
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        def validate(bloomDay, m, k, now):
            count, consecutive = 0, 0
            for bloom in bloomDay:
                if bloom <= now:
                    consecutive += 1
                    if consecutive == k:
                        consecutive = 0
                        count += 1
                else:
                    consecutive = 0
            return count >= m
        latestbloom = 0
        for bloom in bloomDay:
            latestbloom = max(bloom, latestbloom)
        left, right = 0, latestbloom + 1
        while left < right:
            mid = (left + right) // 2
            if validate(bloomDay, m, k, mid):
                right = mid 
            else:
                left = mid + 1
        return -1 if right == latestbloom + 1 else right
```




## 2 排序

![在这里插入图片描述](https://img-blog.csdnimg.cn/6af8703a72f44dea9522685e804a9a24.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)




* 基于比较的排序 
通过比较大小来决定元素间的相对次序 
可以证明时间复杂度下界为O(logN)—— 不可能突破这个复杂度达到更快

* 非比较类排序 
不通过比较大小来决定元素间的相对次序 
时间复杂度受元素的范围以及分布等多种因素影响，不单纯取决于元素数量N



### 2.1 冒泡排序

>**冒泡排序（Bubble Sort）基本思想**：
>>第 i (i = 1，2，… ) 趟排序时从序列中前 n - i + 1 个元素的第 1 个元素开始，==相邻两个元素进行比较，若前者大于后者，两者交换位置，否则不交换==。



* 算法步骤:
  * 比较相邻的元素。如果第一个比第二个大，就交换他们两个。

  * 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对(n和n-1)。这步做完后，最后的元素会是最大的数。

  * 针对所有的元素重复以上的步骤，除了最后一个。

  * 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
![请添加图片描述](https://img-blog.csdnimg.cn/cc02cdf91af7442a952a374a1f2a8747.gif)




---
* 代码实现:

```python
def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
            	arr[j], arr[j + 1] = arr[j + 1], arr[j]
                
    return arr
```



---
* 算法分析:

![请添加图片描述](https://img-blog.csdnimg.cn/e0f32d121b834f64b436bf8d3f6c979b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)




### 2.2 选择排序


> 选择排序（Selection Sort）**基本思想**：
 >>第 i 趟排序从序列的后 n − i + 1 (i = 1, 2, …, n − 1) 个元素中选择一个值最小的元素与该 n - i + 1 个元素的最前面那个元素交换位置，即与整个序列的第 i 个位置上的元素交换位置。如此下去，直到 i == n − 1，排序结束。

==步骤==
* 首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置。

* 再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。

* 重复第二步，直到所有元素均排序完毕。


![请添加图片描述](https://img-blog.csdnimg.cn/ede9a996be04472d9734a861197776af.gif)==python实现==:

```python
def selectionSort(arr):
    for i in range(len(arr) - 1):
        # 记录未排序序列中最小数的索引
        min_i = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_i]:
                min_i = j
        # 如果找到最小数，将 i 位置上元素与最小数位置上元素进行交换
        if i != min_i:
            arr[i], arr[min_i] = arr[min_i], arr[i]
            
    return arr

```



==算法分析==



![请添加图片描述](https://img-blog.csdnimg.cn/6611bee7810b4e93810baa434cd90bea.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)




### 2.3 插入排序



>插入排序（Insertion Sort）基本思想：
>>将整个序列切分为两部分：前 i - 1 个元素是有序序列，后 n - i + 1 个元素是无序序列。每一次排序，将无序序列的首元素，在有序序列中找到相应的位置并插入。


==步骤==
* 将第一待排序序列第一个元素看做一个有序序列，把第二个元素到最后一个元素当成是未排序序列。

* 从头到尾依次扫描未排序序列，将扫描到的每个元素插入有序序列的适当位置。（如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面。）


![请添加图片描述](https://img-blog.csdnimg.cn/8fced408ad124ebcbae10934fad636e3.gif)==python实现==
```python
def insertionSort(a)
	for i in range(1, range(len(a)):  # j从1开始
		j = i 
		temp = a[i]   # 暂存i对应的值
		while j > 0 and a[j-1] > temp:  #当上一个值大于暂存值时, 改变a[j], 且j向右移动
			a[j] = a[j-1]
			j -=  1
		
		a[j] = temp  # 比较完后, 令比较过元素的开头赋予temp   即插入
	return a				
```



==算法分析==

![请添加图片描述](https://img-blog.csdnimg.cn/570c640f17d343c29b8d80a062a59f0f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.4 堆排序

**堆排序(Heap Sort)是对选择排序的优化**

>堆排序（Heap sort）基本思想：
>>借用「堆结构」所设计的排序算法。将数组转化为大顶堆，重复从大顶堆中取出数值最大的节点，并让剩余的堆维持大顶堆性质。

* 堆：符合以下两个条件之一的完全二叉树：

  * 大顶堆：根节点值 ≥ 子节点值。
  * 小顶堆：根节点值 ≤ 子节点值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f25869aa4d514f2c864ba302d00ae2e7.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

---

* ==步骤==:
   * a. 将无需序列构建成一个堆，根据**升序降序需求选择大顶堆或小顶堆**;

   * b.将**堆顶元素与末尾元素交换**，将最大元素"沉"到数组末端;

   * c.重新调整结构，使其满足堆定义，然后继续交换堆顶元素与当前末尾元素，反复执行**调整+交换步骤**，直到整个序列有序。




* ==具体步骤== 出处:<https://www.cnblogs.com/chengxiao/p/6129630.html>


- **步骤一 构造初始堆**。将给定无序序列构造成一个大顶堆（一般升序采用大顶堆，降序采用小顶堆)。		
　　(a).假设给定无序序列结构如下:
　　![在这里插入图片描述](https://img-blog.csdnimg.cn/1bc59b0156174d3c98eb6cd9bdfcb8fb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
     (b) 此时我们从最后一个非叶子结点开始（叶结点自然不用调整，第一个非叶子结点 arr.length/2-1=5/2-1=1，也就是下面的6结点），**从左至右，从下至上进行调整**。
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/ff9a32a4fdda4f01a4f019c10525d5d3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

   (c ).找到第二个非叶节点4，由于[4,9,8]中9元素最大，4和9交换。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/af092672c1664dbfba79033f5b82b12e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


   (d)交换导致了子根[4,5,6]结构混乱，继续调整，[4,5,6]中6最大，交换4和6。
![在这里插入图片描述](https://img-blog.csdnimg.cn/9cac65068ef543f08d013ec4100d21b4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
此时，就将一个**无需序列构造成了一个大顶堆**。

* **步骤二 将堆顶元素与末尾元素进行交换，使末尾元素最大**。然后继续调整堆，再将堆顶元素与末尾元素交换，得到第二大元素。如此反复进行交换、重建、交换。

   (a.将堆顶元素9和末尾元素4进行交换
![在这里插入图片描述](https://img-blog.csdnimg.cn/37aef413afd548379c94aae47b101907.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


   (b.重新调整结构，使其继续满足堆定义:
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/0b73b1d82e0147e3bedfd760d942612c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


   (c.再将堆顶元素8与末尾元素5进行交换，得到第二大元素8.
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/6cdb16ee4b924d1b843fdc4b0c1b36b0.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


   (d 后续过程，继续进行调整，交换，如此反复进行，最终使得整个序列有序:
![在这里插入图片描述](https://img-blog.csdnimg.cn/adea19c88f434e86a7a19dd0ac37946b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

* 动图:

![请添加图片描述](https://img-blog.csdnimg.cn/1fef63300fd049dea5d29a8ecf73169e.gif)



* ==代码实现==

```python
# 调整完全二叉树>>>>大顶堆   
def heapify(arr: [int], index: int, end: int ):
    left = index * 2 + 1                        #index的左右子节点
    right = left + 1
    while left <= end:                          #当index为非子节点时
        max_index = index
        if arr[left] > arr[max_index]:
            max_index = left
        if right <= end and arr[right] > arr[max_index]:
            max_index = right
        if index == max_index:                  #若不用交换，则说明已经交换结束
            break
            
        arr[index],arr[max_index] = arr[max_index],arr[index]
        
        index = max_index                      #继续调整index下一节点
        left = index * 2 + 1
        right = left + 1       

# 初始化大顶堆        
def buildMaxHeap(arr: [int]):
    size = len(arr)      
    for i in range((size - 2)//2, -1, -1):                     # (size-2) // 2 是最后一个非叶节点，叶节点不用调整
        heapify(arr, i, size - 1)
    return arr


def MaxHeapSort(arr: [int]):                                            
    buildMaxHeap(arr)
    size = len(arr)
    for i in range(size):
        arr[0],arr[size - i - 1] = arr[size - i - 1],arr[0]
        heapify(arr, 0, size - i - 2)
        
    return arr
```


* ==算法分析==

![请添加图片描述](https://img-blog.csdnimg.cn/cb2f07717e2b436ba128e744f112d9b1.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)




### 2.5 希尔排序---------插入排序Pro

**希尔排序(ShellSort)是对插入排序的优化——增量分组插入排序**


>希尔排序（Shell Sort）基本思想：
>>将整个序列切按照一定的间隔取值划分为若干个子序列，每个子序列分别进行插入排序。然后逐渐缩小间隔进行下一轮划分子序列和插入排序。直至最后一轮排序间隔为 1，对整个序列进行插入排序。



* 先将整个待排序的记录序列==分割成为若干子序列==分别进行直接插入排序，待整个序列中的==记录"基本有序"时==，再对全体记录进行依次直接插入排序。

* ==步骤==:


  * 首先确定一个元素间隔数 `gap`，然后将参加排序的序列按此间隔数从第 1 个元素开始一次分成若干个子序列，即分别将所有位置相隔为 `gap` 的元素视为一个子序列，在各个子序列中采用某种排序方法进行**插入排序**。
  * 然后减少间隔数，并重新将整个序列按新的间隔数分成若干个子序列，再分别对各个子序列进行排序，如此下去，直到间隔数 `gap = 1`。



![在这里插入图片描述](https://img-blog.csdnimg.cn/5f9c4a8965ff49fd83b7303b1ba0ba53.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



* ==代码实现==

```python
def shellSort(arr):
	size = len(arr)
    gap = size // 2    #初始gap
    
    while gap > 0:      # 直到gap == 	1
        for i in range(gap, size):    #插入排序
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap = gap // 2
    return arr
```

* ==算法分析== 			


![请添加图片描述](https://img-blog.csdnimg.cn/d6165297a15a4e8595366bce8f09a013.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



### 2.6 归并排序

>归并排序（Merge Sort）基本思想：
>>采用经典的分治策略，先递归地将当前序列平均分成两半。然后将有序序列两两合并，最终合并成一个有序序列。


* ==步骤==:
  * 初始时，将待排序序列中的 n 个记录看成 n 个有序子序列（每个子序列总是有序的），每个子序列的长度均为 1。
  * 把当前序列组中有序子序列两两归并，完成一遍之后序列组里的排序序列个数减半，每个子序列的长度加倍。
  * 对长度加倍的有序子序列重复上面的操作，最终得到一个长度为 n 的有序序列。

![请添加图片描述](https://img-blog.csdnimg.cn/795f7e8a2c2848e1914863430e172b65.gif)

----
* ==代码实现==

```python
# 自上而下的递归
def merge(left_arr, right_arr):   #合并左右两个子序列
    arr = []
    while left_arr and right_arr:
        if left_arr[0] <= right_arr[0]:    # 将左右序列 看做栈顶在arr[0]的栈  
            arr.append(left_arr.pop(0))    #左右哪个小哪个进arr[]栈
            
        else:
            arr.append(right_arr.pop(0))
            
    if left_arr:                        #处理多余的元素
        arr.append(left_arr.pop(0))   
        
    if right_arr:
        arr.append(right_arr.pop(0))
        
    return arr


def mergeSort(arr):
    n = len(arr)
    
    if n < 2:
        return arr
    
    mid = n // 2
    left_arr = arr[:mid]
    right_arr = arr[mid:]
    return merge(mergeSort(left_arr),mergeSort(right_arr))   #递归 
```


* ==算法分析==
![请添加图片描述](https://img-blog.csdnimg.cn/e36fd18e62914691a44c34009b3914c9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)





### 2.7  快速排序

>快速排序（Quick Sort）基本思想：
>>通过一趟排序将无序序列分为独立的两个序列，第一个序列的值均比第二个序列的值小。然后递归地排列两个子序列，以达到整个序列有序


* ==步骤==
   * 从数列中挑出一个元素，称为 "基准"（pivot）;

   * 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；

   * 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序


![请添加图片描述](https://img-blog.csdnimg.cn/d3904721eb304e4fa477b79cb241c118.gif)

* ==代码实现==

```python
import random
def partition(arr: [int], low: int, high: int): 
    i = random.randint(low, high)
    arr[i], arr[high] = arr[high], arr[i]
    x = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= x:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)

    return arr
```
关于快排不错的文章:    <https://www.jianshu.com/p/2b2f1f79984e>
* ==算法分析==

![请添加图片描述](https://img-blog.csdnimg.cn/d649806a6345491d8d8f344b2825a5e9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


### 2.8 计数排序


>计数排序（Counting Sort）基本思想：
>>使用一个额外的数组 counts，其中第 i 个元素 counts[i] 是待排序数组 arr 中值等于 i 的元素个数。然后根据数组 counts 来将 arr 中的元素排到正确的位置。

* ==步骤==


  * 找出待排序数组中**最大值元素和最小值元素**。
  * 统计数组中每个值为 i 的元素出现的**次数**，存入数组的第 i 项。
  * 对所有的**计数累加**（从 counts 中的第一个元素开始，每一项和前一项累加）。
  * 反向填充目标数组：将每个元素 i 放在新数组的第 counts[i] 项，每放一个元素就要将 `counts[i] -= 1`。


![请添加图片描述](https://img-blog.csdnimg.cn/3c042542141348b3bcf04d16838fc4a2.gif)


- ==python实现==

```python
def countingSort(arr):
    arr_min, arr_max = min(arr), max(arr)      #计算整数最大最小元素, 并求出在其期间所有元素的的数量
    gap = arr_max - arr_min + 1
    size = len(arr)
    counts = [0 for _ in range(gap)]
    
    
    for num in arr:                       #对每区间每一元素进行计数
        counts[num - arr_min] += 1
    for j in range(1,gap):                #计数后将count[]每一元素替换为类加数量
        counts[j] += counts[j - 1]
        

    res = [0 for _ in range(size)]      #利用额外空间储存排序结果
    for i in range(size - 1, -1, -1):
        res[counts[arr[i] - arr_min] - 1] = arr[i]
        counts[arr[i] - arr_min] -= 1
        
    return res
        
    
arr = [1,2,3,9,6,5,4,8,6,9,2,55] 
countingSort(arr)
```




- ==算法分析==


![请添加图片描述](https://img-blog.csdnimg.cn/edc300312d9f4e8494dd773ccfa315ca.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.9 桶排序

>桶排序（Bucket Sort）基本思想：
>>将未排序的数组分到若干个「桶」中，每个桶的元素再进行单独排序。



- ==步骤==

    将区间划分为 n 个**相同大小的子区间**，每个区间称为一个桶。
    遍历数组，将每个元素装入对应的桶中。
    对每个桶内的元素**单独排序（使用插入、归并、快排等算法）**。
    最后按照顺序将桶内的**元素合并**起来。

![在这里插入图片描述](https://img-blog.csdnimg.cn/db37b09e0e4c4bddb7630af5bef8229f.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1b75db210e9b48d7b9a8da1aa848b067.png)

* ==python实现==
	

```python
def insertionSort(arr):
    for i in range(1, len(arr)):
        temp = arr[i]
        j = i
        while j > 0 and arr[j - 1] > temp:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = temp
        
    return arr
    
    
def bucketSort(arr, bucket_size = 5):
    arr_min, arr_max = min(arr), max(arr)
    bucket_count = (arr_max - arr_min) // bucket_size + 1
    buckets = [[] for _ in range(bucket_count)]
    
    for num in arr:
        buckets[(num - arr_min) // bucket_size].append(num)
        
    res = []
    for bucket in buckets:
        insertionSort(bucket)
        res.extend(bucket)
    
    return res
```


* ==算法分析==


![请添加图片描述](https://img-blog.csdnimg.cn/b4419085af0a4c92a745672c978a1f2e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)





###   2.10 基数排序

>基数排序（Radix Sort）基本思想：
>>将整数按位数切割成不同的数字，然后按每个位数分别比较进行排序。


* ==步骤==
  * 基数排序算法可以采用「最低位优先法（Least Significant Digit First）」或者「最高位优先法（Most Significant Digit first）」。最常用的是「最低位优先法」。
下面我们以最低位优先法为例:。

     *  遍历数组元素，获取数组最大值元素，并取得位数。
     *  以个位元素为索引，对数组元素排序。
     *  合并数组。
     *  之后依次以十位，百位，…，直到最大值元素的最高位处值为索引，进行排序，并合并数组，最终完成排序。
![请添加图片描述](https://img-blog.csdnimg.cn/25cc26ddfc7042bd89d468dc8fa1833c.gif)

* ==python==

```python
def radixSort(arr):
    size = len(str(max(arr)))   #求最大元素的位数
    
    for i in range(size):     #排序各位数大小
        buckets = [[] for _ in range(10)]
        for num in arr:
            buckets[num // (10**i) % 10].append(num)
    	arr.clear()
    	for bucket in buckets:
        	for num in bucket:
                arr.append(num)
            
    return arr
```
![请添加图片描述](https://img-blog.csdnimg.cn/cb67d70fdd3344589b64b9daa0440c68.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



**插入、冒泡、归并、计数、基数和桶排序是稳定的 
选择\希尔\快速、堆排序是不稳定**


![在这里插入图片描述](https://img-blog.csdnimg.cn/cce17bba8b484de1ad64716588c04c70.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.11 相关题目



1. [1122 . 数组的相对排序](https://leetcode-cn.com/problems/relative-sort-array/)



* ==思路==: 先建立`arr2`**数值到下标的映射** >>>  自定义排序

```cpp
class Solution {
public:
    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        unordered_map<int, int> rank;
        for (int i = 0; i < arr2.size(); ++i) {
            rank[arr2[i]] = i;
        }
        sort(arr1.begin(), arr1.end(), [&](int x, int y) {
            if (rank.count(x)) {
                return rank.count(y) ? rank[x] < rank[y] : true;
            }
            else {
                return rank.count(y) ? false : x < y;
            }
        });
        return arr1;
    }
}
```

自定义函数  ---------------- O(nlogn)
python中考虑sort的key解决:


```python
return sorted(arr1, key = lambda x:(0, arr2.index(x)) if x in arr2 else (1, x))
```


* ==思路二==: 计数 -------------- O(n)


```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        ans, count = [], [0]*1001
        for arr in arr1:
            count[arr] += 1
        for arr in arr2:
            while count[arr] > 0:
                ans.append(arr)
                count[arr] -= 1

        for arr in range(1001):
            while count[arr] > 0:
                ans.append(arr)
                count[arr] -= 1
        return ans
```





2. [56 . 合并区间](https://leetcode-cn.com/problems/merge-intervals/)



* 法一:  自定义排序 + 区间取舍
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x:x[0])
        ans, start, farthest = [], -1, -1
        for interval in intervals:
            left, right = interval[0], interval[1]
            if left <= farthest:   #如果下一区间与之间有重叠
                farthest =  max(farthest, right)
            else:
                if farthest != -1:
                    ans.append([start, farthest]) 
                start = left
                farthest = right
        ans.append([start, farthest])    
        return ans
```

* ==思路二==: 差分

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        ans, events = [], []
        for interval in intervals:
            events.append([interval[0], 1])
            events.append([interval[1] + 1, -1])
        events.sort(key = lambda x:(x[0],x[1]))
        covering = 0
        for event in events:
            if covering == 0:
                start = event[0]
            covering += event[1]
            if covering == 0:
                ans.append([start, event[0] - 1])
        return ans

```




3. [215 . 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

思路: 快排 >>>> partition左右侧数量



```python

class Solution:
    import random
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return self.quickSort(nums, 0, len(nums) - 1, len(nums) - k)

    def quickSort(self, arr, l, r, index):
        if l >= r:
            return arr[l]
        pivot = self.partition(arr, l, r)
        if index <= pivot:
            return self.quickSort(arr, l, pivot, index)
        else:
            return self.quickSort(arr,pivot + 1, r, index)

    def partition(self, a, l, r):
        pivot = random.randint(l, r)
        pivotVal = a[pivot]
        while l <= r:
            while a[l] < pivotVal:
                l += 1
            while a[r] > pivotVal:
                r -= 1
            if l == r:break
            if l < r:
                a[l], a[r] = a[r], a[l]
                l += 1
                r -= 1
        return r

```


## 作业
1.  [1011 . 在 D 天内送达包裹的能力](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/)
  * 思路:  船最小的运载能力，最少也要等于或大于最重的那件包裹，即 max(weights)。最多的话，可以一次性将所有包裹运完，即 sum(weights)。船的运载能力介于 `[max(weights), sum(weights)]` 之间。 >>>>  同时先计算运载能力mid = (high + low) // 2 的运输天数, 再与之D比较进而缩小区间:

```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        low, high = max(weights), sum(weights)
        while low < high:
            mid = low + (high - low) // 2
            # 计算载重为mid 需要多少天运完
            cur = 0         # 目前天数的重量
            day_mid = 1     # 目前天数
            for weight in weights:
                if weight + cur > mid:
                    day_mid += 1
                    cur = 0
                cur += weight


            if day_mid > days:  #如果天数超了, 则增加载重
                low = mid + 1

            else:
                high = mid
        return low
```



2. [154 . 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)


```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left)//2
            if nums[mid] < nums[right]:
                right = mid
            elif nums[mid] == nums[right]:
                right -= 1
            else:
                left = mid + 1
        return nums[right]
```

## 参考资料
1.<https://algo.itcharge.cn>  算法通关手册（LeetCode）
