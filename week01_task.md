# 一、数组, 链表, 栈及队列

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----

- [And a table of contents](#and-a-table-of-contents)
- [On the right](#on-the-right)
    + [<font color=red face="微软雅黑">来源</font>](#-font-color-red-face------------font-)
- [1 数组](#1---)
  * [1.1 理论](#11---)
    + [1.1.1 概念及总结](#111------)
    + [1.1.2 数组在C++, Java, Python中的表现](#112----c----java--python----)
    + [1.1.3 数组的操作](#113------)
  * [1.2 例题实战](#12-----)
  * [1.3  设计变长数组---------面试常问](#13---------------------)
- [2 链表](#2---)
  * [2.1 链表基础知识](#21-------)
  * [2.2 链表基础题目](#22-------)
  * [2.3 链表翻转及链表与数组映射题目](#23---------------)
- [3 栈和队列](#3-----)
  * [3.1 栈](#31--)
  * [3.2 队列](#32---)
  * [3.3 相关题目](#33-----)
    + [3.3.1 20 . 有效的括号](#331-20-------)
    + [3.3.2 155 . 最小栈](#332-155-----)
  * [3.4 表达式求值问题](#34--------)
    + [3.4.1 150 . ](#341-150----------)
    + [3.4.2 227 . 基本计逆波兰表达式求值算器 II](#342-227--------ii)
- [4 本周作业](#4-----)
  * [4.1 66加一](#41-66--)
  * [4.2 21 . 合并两个有序链表](#42-21----------)
- [参考资料](#----)




---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 数组
### 1.1 理论
#### 1.1.1 概念及总结
>**数组（Array**）：一种线性表数据结构。它使用一组连续的内存空间，来存储一组具有相同类型的数据。


→     <font color=red face="微软雅黑">==相同类型==的数据元素构成的==有序集合==.</font>
→  注意定义里的 -----  ==一组连续的内存空间==

* 特点: 支持**随机访问**
* 关键: **索引**和**寻址**
---

一维数组示例 : 
![在这里插入图片描述](https://img-blog.csdnimg.cn/eadc820c4bd245e0b083594456f597b7.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----
* 数组和链表操作速度对比:

![](https://img-blog.csdnimg.cn/2aeaea2be77245fe9acd980814a15e77.png)
#### 1.1.2 数组在C++, Java, Python中的表现
•   `C++: int a[100]; `
•  `Java: int[] a = new int[100]; `
•   `Python: a =[]`                  `#列表`

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

---
  2. **Java**
  Java 中的数组也是存储相同类型数据的，但所使用的内存空间却**不一定是连续（多维数组中）**。且如果是多维数组，其嵌套数组的长度也可以不同。例如：
  ```java
  int[][] arr = new int[3][]{ {1,2,3}, {4,5}, {6,7,8,9}};
  ```
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/243b136fd4664105833165aab98c2152.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

如上图, 在Java中，数组都是引用实体变量，==呈树形结构，每一个叶子节点之间毫无关系，只有引用关系，每一个引用变量只引用一个实体==。

---
  3. Python
  	原生 Python 中其实没有数组的概念，而是使用了类似 Java 中的 ArrayList 容器类数据结构，叫做列表。通常我们把列表来作为 Python 中的数组使用。Python 中列表==存储的数据类型可以不一致，数组长度也可以不一致==。例如：
```python
  	arr = ['python', 'java', ['asp', 'php'], 'c']
 ```

---

#### 1.1.3 数组的操作
* 即==增查改删==.

1. 访问及查找
   *  访问数组中第 `i` 个元素：只需要检查` i `的范围是否在合法的范围区间, **时间复杂度为O(1)**
   * 查找数组中元素值为 val 的位置：在数组无序的情况下，只能通过将 val 与数组中的数据元素逐一对比的方式进行检索，也称为**线性查找**------------**时间复杂度为O(n)** 。
2.  插入
    * ==尾部插入==:如果数组尾部容量不满，**则直接把 val 放在数组尾部的空闲位置**，并更新数组的元素计数值。如果数组**容量满了，则插入失败**。
    不过，`Python` 中的 `list` 做了其他处理，当数组容量满了，则会开辟新的空间进行插入。在尾部插入元素的操作不依赖数组个数，**其时间复杂度为O(1)** 。
    python中. 使用`append`即可:
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
    *  删除数组**尾部**-----------------将元素计数值减一即可, 时间复杂度为 O(1)。------python中使用`pop`
    * 删除数组**第 i 个位置上的元素**---------先检查下标 i 是否合法，即 `o <= i <= len(nums) - 1`。如果下标合法，则将第 i + 1 个位置到第 len(nums) - 1 位置上的元素依次向**左移动**。删除后修改数组的元素计数值。删除中间位置元素的操作同样涉及移动元素，而移动元素的操作次数跟元素个数有关，因此删除中间元素的最坏和平均时间复杂度都是O(n)-------------同样使用`pop`

![在这里插入图片描述](https://img-blog.csdnimg.cn/e96ad183d3de4f41ba4afc5a8ed927fb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)         

   3. 基于**条件删除元素** :------------------**通过循环检查元素，查找到元素后将其删除**。删除多个元素操作中涉及到的**多次移动元素操作**，可以通过算法改进，将多趟移动元素操作转变为一趟移动元素，从而将时间复杂度降低为O(n) 。一般而言，这类删除操作都是线性时间操作，时间复杂度为O(n) 。---------------`remove`


```python
arr = [0, 5, 2, 3, 7, 1, 6]
i = 3
arr.remove(5)
print(arr)
```
---





---

### 1.2 例题实战

 
1.   26 -  [去重](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)
   --   重点  : 原地删除元素不占用新的内存
   
   -- 模型:  给定有序数组 nums ，**原地**删除重复出现的元素  ==→→==     **for循环, if条件选择出元素**设置过滤器, 并写边界条件. 

python解法如下
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = 0
        for i in range(len(nums)):
            if (nums[i] != nums[i-1]) or i == 0:  #不等于上一个元素或为第一个元素
                nums[n] = nums[i]
                n += 1

        return n
```
---

![在这里插入图片描述](https://img-blog.csdnimg.cn/2cd009d548614046911fa6819d2c624d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


- ==知识点笔记==:
  [1]. **为什么返回数值是整数，但输出的答案是数组呢?**
请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
你可以想象内部操作如下:
```cpp
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```


2. 283- [移动零](https://leetcode-cn.com/problems/move-zeroes/)
  -- 重点: 将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序

   -- 模型:  保序操作非0元素  ==→→==     **`for`循环, `if`条件选择出元素**设置过滤器(非0), 并写边界条件 , 后续`while`补0


```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        count = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[count] = nums[i]
                count += 1
        
        while count < len(nums):
            nums[count] = 0
            count += 1
```

3. [88 . 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

   -- 重点: **合并** nums2 到 nums1 中， **非递减顺序** 排列;  合并**存储在数组 nums1 中**  ;  时间复杂度为 `O(m + n)`

   -- 笔记:  使用双指针  ==→→== 两个指针都到尾部结束**且需要额外空间** 因为如果直接合并到数组==nums1中, nums1中的元素可能会在取出之前被覆盖==.   ==→→== **因此引入倒序双指针, 由于num1后半有n个0用来存符合`if`条件值不会被覆盖:**

```python
	class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        count_1 = m-1
        count_2 = n-1
        for p in range(m + n-1, -1, -1):  #倒序nums1[:]并准备筛选值
            if count_2 < 0 or (nums1[count_1] >= nums2[count_2] and count_1 >= 0 ):
                # 通过if筛选出连个数组较大项  或者  nums2数组出界时  选择nums1的元素
                nums1[p] = nums1[count_1]
                count_1 -= 1

            else:
                nums1[p] = nums2[count_2]
                count_2 -= 1
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/f54e4e4ca10641a381b9b44839973551.png)



### 1.3  设计变长数组---------面试常问
- 变长数组是==数组大小待定的数组==，C语言中结构体的最后一个元素可以是大小未知的数组，也就是所谓的0长度。	
* C++: vector 
* Java: ArrayList 
* Python: list

**不需要指定数组长度**

---
* 如何实现一个变长数组？  ----------考虑4个因素:
  * 支持**索引与随机访问** 
  * 分配多长的**连续空间**？ 
  * **空间不够**怎么办？   --- 输入可能超出
  * **空间剩余很多**如何回收?  ---- pop过多

---

* ==建立这些接口供用户使用==
[1] 新建数组
[2] `get[i]`函数  ---- 获取`index[i]`的值;   `set(i,val)` ----- 等价于`a[i] = val`赋值操作
    并对`get[i], set(i,val)`作边界检查
 [3] push_back(val) ----------末尾插入元素
 [4] pop_back  ------------  在末尾删除元素

---

* ==实现方法---简易方法==
初始：**空数组,  分配常数空间,  记录实际长度(size)和容量(capacity)**
Push back:**若空间不够，重新申请2倍大小的连续空间，拷贝到新空间，释放旧空间**  ----------- 为了保证用户使用空间是连续的, 类似内存交还
Pop back:**若空间利用率(size/capacity)不到25%,释放一半的空间**  -------不能设置为50%,可能导致频繁的扩容拷贝


* ==均摊O(1)== -----------较佳
•在空数组中连续插入n个元素、总插入/拷贝次数为n + n/2 + n/4 + ... < 2n 
•一次扩容到下一次释放，至少需要再删除n-2n*0.25 = 0.5n次  --------触发收缩pop back





## 2 链表
### 2.1 链表基础知识
> ==链表==是一种物理存储**结构上非连续，非顺序的存储结构**，数据元素的逻辑顺序是通过链表中的**指针链接次序**实现的。


* ==链表结构==:  一般两种---------无头单链  和  有头双链

![在这里插入图片描述](https://img-blog.csdnimg.cn/b131661d06a04296b366789b532d8a81.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/7dd0597ea71d40d49b8ea3eaa31d1836.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
* 单双区别:
    * 单链表的每一个节点中只有指向下一个结点的指针，不能进行回溯，适用于节点的增加和删除。
    * 双链表的每一个节点给中既有指向下一个结点的指针，也有指向上一个结点的指针，可以快速的找到当前节点的前一个节点，适用于需要双向查找节点值的情况。


---
==链表的操作==

* 插入 -----O(1)
![在这里插入图片描述](https://img-blog.csdnimg.cn/ff330593e3c841d888dac8ba44a791a6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)* ==删除== --------------O(1)
![在这里插入图片描述](https://img-blog.csdnimg.cn/e1b24f055c5b4556ba67669ee2699b45.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
----
* ==链表和数组的区别==：
  * 数组静态分配内存，链表动态分配内存。
  *  数组在内存中是连续的，链表是不连续的。
  *  数组利用下标定位，查找的时间复杂度是O(1)，链表通过遍历定位元素，查找的时间复杂度是O(N)。
  * 数组插入和删除需要移动其他元素，时间复杂度是O(N)，链表的插入或删除不需要移动其他元素，时间复杂度是O(1)。

---
==基本元素==：
* 节点：每个节点有两个部分，左边称为值域，存放用户数据；右边部分称为指针域，用来存放指向下一个元素的指针。
* `head`:head节点永远指向第一个节点
* `tail`: tail永远指向最后一个节点
* `None`:链表中最后一个节点的指针域为None值




---


 ==建立链表==

1 .分配内存
2.存储数据
3.处理指针
* 建立一个只有一个节点的链表的函数，示例如下
```cpp
nodeptr_t NewLinkedList(int val) {
    // 建立第一个节点
    nodeptr_t head = NULL
    head = malloc(sizeof(node_t));

    head->data = val;
    head->next = NULL;
    return head;
}
```

python

```python
class Node:
    def __init__(self,data = None, next = None):
        self.data = data
        self.next = next



node1 = Node(1)   #创建节点
node2 = Node(2)
node3 = Node(3)  


node1.next = node2   # 确定节点关系
node2.next = node3
```


### 2.2 链表基础题目

1.[206 . 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)


==法一==  双指针(pre cur)遍历迭代
![请添加图片描述](https://img-blog.csdnimg.cn/d8ee29119cc64820afcbbcc30722cfc7.gif)


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None     # head指向空  
        cur = head  

        while cur:
            tmp = cur.next   #暂时存储cur的下一节点, 接下来cur.next需要先被赋值
            cur.next = pre   # 将cur 指向 pre

           # 两个指针向后移动
            pre = cur 
            cur = tmp
        return pre
```
* 两个指针，第一个指针叫 pre，最初是指向 null 的。
第二个指针 cur 指向 head，然后不断遍历 cur。
每次迭代到 cur，都将 cur 的 next 指向 pre，然后 pre 和 cur 前进一位。
都迭代完了(cur 变成 null 了)，pre 就是最后一个节点了。

==法二==  递归



* 递归的两个条件：

   * 终止条件是当前节点或者下一个节点==null
   * 在函数内部，改变节点的指向，也就是 head 的下一个节点指向 head 递归函数那句
```python
head.next.next = head
```

![请添加图片描述](https://img-blog.csdnimg.cn/98a249b9529b4a8782900a9810d3b384.gif)

```python
class Solution(object):
	def reverseList(self, head):
		"""
		:type head: ListNode
		:rtype: ListNode
		"""
		# 递归终止条件是当前为空，或者下一个节点为空
		if(head==None or head.next==None):
			return head
		# 这里的cur就是最后一个节点
		cur = self.reverseList(head.next)
		# 这里请配合动画演示理解
		# 如果链表是 1->2->3->4->5，那么此时的cur就是5
		# 而head是4，head的下一个是5，下下一个是空
		# 所以head.next.next 就是5->4
		head.next.next = head
		# 防止链表循环，需要将head.next设置为空
		head.next = None
		# 每层递归函数都返回cur，也就是最后一个节点
		return cur

```


### 2.3 链表翻转及链表与数组映射题目
2.[25 . K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

==笔记==:  根据题意: 
  1. 先分组(往后走k-1步找到一组);  一组开头以head  结尾以end
  2. 一组内部翻转链表 ; 
  3. 更新每一组 有前后之间的边



---


3. 136.[邻值查找](https://www.acwing.com/problem/content/description/138/)



* 先对**整个序列**排序
* 再倒着查看每一元素的**前继后续**, 相同差值则取**前继** →  取完删除该元素  →倒数第二元素重复操作


==为了删除方便== →  考虑到使用链表   →   利用双向链表(存顺序关系)和数组(存指针)建立映射	

---

* ==关键点总结==
  * "索引〃的灵活性一按下标/按值 
  * 不同"索引〃的数据结构之间建立〃映射〃关系 
  * 倒序考虑问题



---

## 3 栈和队列

### 3.1 栈
>**只允许在一端进行插入或删除操作的线性表**。
>>首先，栈是一种**线性表**，但限定这种线性表只能在**某一段进行插入和删除操作**。


→→→→   栈是一种<font color=red face="微软雅黑">只能从表的一端存取数据且遵循 "先进后出" 原则</font>的线性存储结构
* ==存储结构==
	![在这里插入图片描述](https://img-blog.csdnimg.cn/3d2119bc51964796849f105de742cb4c.png)
包含
 **栈顶（Top**）：线性表允许进行插入和删除的一端。
**栈底（Bottom**）：固定的，不允许进行插入和删除的另一端。
**空栈**：不含任何元素。


---


* ==两种实现方式==

  * **顺序栈**：采用顺序存储结构可以模拟栈存储数据的特点，从而实现栈存储结构；
  * **链栈**：采用链式存储结构实现栈结构
* 区别:  物理存储位置区别--------------顺序栈底层采用的是**数组**，链栈底层采用的是**链表**




### 3.2 队列

 * 与栈结构不同的是，队列的两端都"开口"，要求数据只能从一端进，从另一端出

![在这里插入图片描述](https://img-blog.csdnimg.cn/86053f9a911f4b3b8447a04af33928d6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
==遵循 "先进先出" 的原则==


* 实现

  * **顺序队列**：在**顺序表**的基础上实现的队列结构；
  * **链队列**：在**链表**的基础上实现的队列结构；


----


* ==双端队列==
 >双端队列又名double ended queue，简称deque，双端队列没有队列和栈这样的限制级，它允许两端进行入队和出队操作，也就是说元素可以从队头出队和入队，也可以从队尾出队和入队。


![在这里插入图片描述](https://img-blog.csdnimg.cn/e9f1e88b9c704e5eaf8a5b818f999bd3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

---
* ==优先队列==
>优先队列是基于二叉堆实现, 优先指的是按==某种优先级==优先出列

  * "优先级"可以是自己定义的一个元素属性 
  * 许多数据结构都可以用来实现优先队列，例如二叉堆、二叉平衡树等





----


* ==时间复杂度==

**栈、队列** 
  * Push (入栈、入队)：O(1) 
* Pop (出栈、出队)：O(1) 
*  Access (访问栈顶、访问队头)：O(1) 

**双端队列** 
* 队头、队尾的插入、删除、访问也都是O(1) 
	
**优先队列** 
* 访问最值：O(1) 
* 插入：一般是◦ (logN),-些高级数据结构可以做到O(1) 
* 取最值:  O(logN)


---
* python中
栈、队列、双端队列可以用 list 实现
优先队列可以用 heapq 库
### 3.3 相关题目


#### 3.3.1 20 . 有效的括号
<https://leetcode-cn.com/problems/valid-parentheses/>


* ==思路==:   最近括号需要左或右与之匹配>>>>>>>>>**最近相关性**>>>>>>先进后出>>>栈来考虑	>>>匹配成功>>>>出栈
      >>>匹配不超过>>>>>不规范


python实现:
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []  #python用列表建立栈
        dic = { ')' : '(',
                ']' : '[',
                '}' : '{'}    #建立括号对应哈希, 并使值(左括号)先入栈

        for i in s:
            if stack and i in dic:     # 栈栈非空 且为 右扩则开始判断
                if stack[-1] == dic[i]:  
                    stack.pop()        #若左右相匹配则消去

                else:
                    return False

            else:                #若栈空 或者 左括号则入栈
                stack.append(i)

        return not stack        #判断最后存活下的栈是否为空
```



#### 3.3.2 155 . 最小栈
<https://leetcode-cn.com/problems/min-stack/>

* ==思路==: 用**额外空间(O(n))记录栈历史(前缀)最小值**>>>>>两个栈>>>>防止pop丢失最小值



* python>>>>>>>>>利用一个栈，这个栈同时保存的是**每个数字 x 进栈的时候的值 与 插入该值后的栈内最小值**。即每次新元素 x 入栈的时候保存一个元组：（==当前值 x，栈内最小值==）。


```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if not self.stack:
            self.stack.append((x, x))
        else:
            self.stack.append((x, min(x, self.stack[-1][1])))
        

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()
        

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]
        

    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]

	
```
### 3.4 表达式求值问题


* 前缀表达式 
   * 形如"opAB",其中op是一个运算符，A,B是另外两个前缀表达式 ・例如:*3 + 12 
   * 又称波兰式 
* 后缀表达式 
   * 形如"ABop" 
   *   12+3* 
   * 又称逆波兰式 
 * 中缀表达式 
   *   3*(1 +2)



#### 3.4.1 150 . 逆波兰表达式求值
<https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/>

==思路==  逆波兰>>>>>>>>>>>>最近相关性>>>>>>>>>用栈






#### 3.4.2 227 . 基本计算器 II
<https://leetcode-cn.com/problems/basic-calculator-ii/>

==思路==:  考虑存在运算符优先级>>>>>>>>>>运算符和数字分开存栈>>>>>>>中缀转成后缀形式





## 4 本周作业
### 4.1 66加一

<https://leetcode-cn.com/problems/plus-one/>
* 法一--------------直接想到的----------转换成int+1  再转回列表

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        dig = int(''.join(str(i) for i in digits)) + 1
        return [int(i) for i in list(str(dig))]
```



* ==法二==-----------首位补0 + 尾巴加1 循环判断尾巴是否为9

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


### 4.2 21 . 合并两个有序链表

<https://leetcode-cn.com/problems/merge-two-sorted-lists/>



==法一==-------标准做法----移动一个指针



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        cur = ListNode()
        temp = cur        # 保存头指针，移动的是cur指针

        while l1 and l2:          #根据条件决定cur下一节点
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next

            else:
                cur.next = l2
                l2 = l2.next   
            cur = cur.next

        if l1:             #处理多余的节点
            cur.next = l1

        if l2:
            cur.next = l2

        return temp.next
```




==法二==>>>>>>>>>>>>>>>>>>>>==递归==

* 终止条件：当两个链表都为空时，表示我们对链表已合并完成。
* 如何递归：我们判断 l1 和 l2 头结点哪个更小，然后较小结点的 next 指针指向其余结点的合并结果。（调用递归）

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2  # 终止条件，直到两个链表都空
        if not l2: return l1
        if l1.val <= l2.val:  # 递归调用
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/d15d82433bf948edb91b1c1ac64c178d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



## 参考资料


1. <https://zhuanlan.zhihu.com/p/105962783>-----什么是数组？
2. <https://www.runoob.com/cprogramming/c-arrays.html> C数组
3. <https://github.com/itcharge/LeetCode-Py>Datewhale--31期算法leetcode刷题
4. <https://www.eet-china.com/mp/a55099.html>变长数组
5. <https://blog.csdn.net/Shuffle_Ts/article/details/95055467>链表
6. <https://zhuanlan.zhihu.com/p/346164833>栈
7. <http://data.biancheng.net/view/169.html>栈及其特点
8. <http://data.biancheng.net/view/172.html>队列
