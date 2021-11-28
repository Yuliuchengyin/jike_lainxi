# 二、单调栈, 单调队列,哈希表及集合

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----

@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 单调栈, 单调队列

>单调栈就是 **栈内元素单调递增或者单调递减** 的栈，单调栈只能在栈顶操作。


单调栈  >>> 一般只处理 `Next Greater Element`问题 >>>> **达到线性复杂度**
* 单调栈, 单调队列并非一种数据结构, 而是利用单调性排除冗余  >>>>  **对算法进行优化**


单调栈的维护是 O(n) 级的时间复杂度，因为所有元素只会进入栈一次，并且出栈后再也不会进栈了。

单调栈的性质：

1.单调栈里的元素具有单调性

2.元素加入栈前，会在**栈顶端把破坏栈单调性的元素都删除**  ---- 直到满足单调

3.使用单调栈**可以找到元素向左遍历第一个比他小的元素**，也可以找到**元素向左遍历第一个比他大的元素。**




---
* ==单调栈题目代码套路：== 
•  for 每个元素 
•      while （栈顶与新元素不满足单调性）{弹栈，更新答案，累加"宽度"} 
•   入栈

### 1.1 例题1   84 . 柱状图中最大的矩形
<https://leetcode-cn.com/problems/largest-rectangle-in-histogram/>

* 解题思路:  
1. 考虑如果矩形**高度单调递增**, 每一个高度对应最大矩形面积 >>>> 矩形**高度向右扩展宽度**
2. 若出现破坏单调性 >>>> 最后一个**矩形(栈顶) 高度确定** >>> 其对应最大矩形面积确定 >>> 累加宽度(宽度留给之前矩形计算面积) >>>> 栈顶(for到目前最大高度的矩形) 出栈  >>> 直到矩形的高度满足单调位置  >>>>    go  on  for
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/b75403cf682f4bd4982d13165c8ec3c2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


*  ==法一== ------------- 老师答案的python写法:

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack, ans, heights = [], 0, heights + [0]

        for i in heights:
            accumulated_wide = 0
            while stack and (i < stack[-1][1]):   # stack[:][1]存高度 stack[:][0]存宽度
                accumulated_wide += stack[-1][0]
                ans = max(ans, accumulated_wide * stack[-1][1])
                stack.pop()
            stack.append([accumulated_wide + 1, i])

        return ans
```



* ==法二==    ----------- 同样单调栈 -------- 通过利用python栈(列表)下标记录宽度, 并且左右加哨兵


```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        heights = [0] + heights + [0]
        res = 0
        for i in range(len(heights)):
            #print(stack)
            while stack and heights[stack[-1]] > heights[i]:
                tmp = stack.pop()
                res = max(res, (i - stack[-1] - 1) * heights[tmp])
            stack.append(i)
        return res

```
### 1.2 例题2   42 . 接雨水

<https://leetcode-cn.com/problems/trapping-rain-water/>

* ==思路== :  雨水两侧为空则填充,  碰到矩形则停止 >>>>>  递减单调栈


![在这里插入图片描述](https://img-blog.csdnimg.cn/18f52570e4d244a4b27754b2ad91111a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

易知
水高度:  `min(height[left],height[i])−height[top]`
水的宽度:  `i-stack[left] - 1`


* ==方法一==-----------单调栈


```python
class Solution:
    def trap(self, height: List[int]) -> int:
        stack, ans, size = [], 0, len(height)
        for i in range(size):
            while stack and height[i] > height[stack[-1]]:
                temp = height[stack[-1]]
                stack.pop()
                if not stack: break   #如栈空停止while
                ans += (min(height[i], height[stack[-1]]) - temp)*(i - stack[-1] - 1)
            stack.append(i)
        return ans
```



* ==方法二==-------------动态规划-----------韦恩图
   *  i 向左遍历, 计算目前最大高度包围面积----S1
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/626a56c56f0646a386723a40dbf75df6.png)
   *  j 向右遍历, 同样计算最大高度包围面积----S2
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/c5f3479d0e7f46b7b7becd510ea450af.png)
S1 + S2 = 全集 ，则：重复面积 = 柱子面积 + 积水面积   >>>>  积水面积 = S1 + S2 - 矩形面积 - 柱子面积

![在这里插入图片描述](https://img-blog.csdnimg.cn/cdc3789ba613425981f6bbe0d78b06ac.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_14,color_FFFFFF,t_70,g_se,x_16)

```python
	class Solution:
    def trap(self, height: List[int]) -> int:
        max_1, max_2, S1, S2 = 0, 0, 0, 0
        size = len(height)
        for i in range(size):
            if height[i] > max_1:
                max_1 = height[i]
            
            if height[size - 1 - i] > max_2:
                max_2 = height[size - 1 - i]
            S1 += max_1 
            S2 += max_2 

        return (S1 + S2 - sum(height) - max_2 * size)
```


### 1.3 例题3   239 . 滑动窗口最大值

<https://leetcode-cn.com/problems/largest-rectangle-in-histogram/>


* 滑动窗口 >>>> 一边进另一边出 >>>> 队列
* 满足递减时可一直入队列, 并且下标为0即是最大值>>>> 递减的单调队列
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        stack, size, ans = collections.deque(), len(nums),[]
        for i in range(size):
            if stack and i - stack[0] >= k:            #队列头一个元素出队列
                stack.popleft()
            while stack and nums[i] > nums[stack[-1]]:  # 不满足递减将目前队列最后元素淡出
                stack.pop()
            stack.append(i)
            if i >= k - 1:                            # 当i大于k-1时, 开始输入答案
                ans.append(nums[stack[0]])            
        return ans
```


* ==注意==/:  pythoon中使用标准库中`deque`来构造队列, 比`List`快20倍左右:


![在这里插入图片描述](https://img-blog.csdnimg.cn/423170d2a8194d3792b4606f883525e9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



## 2 哈希表
### 2.1 基础知识

>**哈希表(Hash table**是一种根据关键码去寻找值的数据映射结构，该结构通过把**关键码映射**的位置去寻找存放值的地方

哈希表由两部分组成 
*  **_个数据结构**、通常是链表、数组 
*   **Hash函数**，输入"关键码"(key),返回数据结构的索引

>>>**hash函数就是根据key计算出应该存储地址的位置，而哈希表是基于哈希函数建立的一种查找表**
---

* **Hash函数**设计的考虑因素
1.计算散列地址所需要的时间（即hash函数本身不要太复杂）
2.关键字的长度
3.表长
4.关键字分布是否均匀，是否有规律可循
5.设计的hash函数在满足以上条件的情况下尽量减少冲突



![在这里插入图片描述](https://img-blog.csdnimg.cn/35f679fe5ea140d8a7acfbb65fbbb524.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.2 哈希冲突

> 即不同key值产生相同的地址，H（key1）=H（key2）

* 把复杂信息映射到小的值域，发生碰撞是不可避免的 
好的**Hash函数**可以减少碰撞发生的几率，让数据尽可能地均衡分布

* 哈希冲突的最常见解决方案-----------------**链地址法**
   *  Hash函数依然用于计算数组下标 
   *  数组的每个位置存储一个链表的表头指针(我们称它为表头数组)
   *  每个链表保存具有同样Hash值的数据

![在这里插入图片描述](https://img-blog.csdnimg.cn/a9d9ef96a5ad4b78b147d16fd73f4095.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_10,color_FFFFFF,t_70,g_se,x_16)


* 完整结构图:
![在这里插入图片描述](https://img-blog.csdnimg.cn/21bc71683f48452083aa2bbb3990b862.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.3 应用及时间复杂度

* 电话号码簿 
* 用户信息表 
* 缓存(LRU Cache) 
* 键值对存储(Redis)


---
* 期望：**插入、查询、删除O(1**) >>>数据分布比较均衡时 
* 最坏：**插入、查询、删除O(n)** >>>>数据全部被映射为相同的Hash值时



### 2.4 相关例题
#### 2.4.1 [874 . 模拟行走机器人](https://leetcode-cn.com/problems/walking-robot-simulation/)

* 思路:   开二维数组会导致内存爆炸>>>>>>>不可取>>>>>>坐标做哈希(用无序集合)

==技巧==:  顺时针设定方向, 并设定方向数组: N = 0 E = 1 S =2 W=3;  dx = {0, 1, 0, -1}  dy = {1, 0, -1, 0}


```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:

        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        x = y = di = 0
        obstacleSet = set(map(tuple, obstacles))
        ans = 0

        for cmd in commands:
            if cmd == -2:  #left
                di = (di - 1) % 4
            elif cmd == -1:  #right
                di = (di + 1) % 4
            else:
                for k in range(cmd):
                    if (x+dx[di], y+dy[di]) not in obstacleSet:
                        x += dx[di]
                        y += dy[di]
                        ans = max(ans, x*x + y*y)

        return ans
```














## 3 本周作业

### 3.1 [697 . 数组的度](https://leetcode-cn.com/problems/degree-of-an-array/)
* ==思路==: 对出现元素及次数, 开始下标以及结束下标建立哈希>>> 遍历数组记录[次数, 开始下标, 结束下标]>>>>>>>>>>对比次数计算度>>>>>>>计算最小距离



```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        dic = {}             #设置字典记录元素始终位置及出现次数
        for i in range(len(nums)):
            if nums[i] in dic:
                dic[nums[i]][0] += 1
                dic[nums[i]][2] = i
            else:
                dic[nums[i]] = [1,i,i]
         
        degree, min_len = 0, 50000
        for count in dic.values():   #计算degree
            if count[0] > degree:
                degree = count[0]

        for i, l, r in dic.values():  #寻得最小宽度
            if i == degree:
                min_len = min(min_len, r - l + 1)
        return min_len
```



### 3.1 [811 . 子域名访问计数](https://leetcode-cn.com/problems/subdomain-visit-count/)

* 思路:   对每一个子域名及其访问次数建立哈希即可


```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        ans = {}
        for i in cpdomains:
            num, doms = i.split()
            size = len(doms.split('.'))

            for j in range(size):
                dom = doms.split('.', j)[-1]
                if dom in ans:
                    ans[dom] += int(num)
                else:
                    ans[dom] = int(num)

        return [str(v)+' '+k for k, v in ans.items()]
```




## 参考资料 
1. <https://www.cnblogs.com/RioTian/p/13462825.html> 单调栈笔记
2. <https://blog.csdn.net/liujian20150808/article/details/50752861>单调栈小结
3. <https://blog.csdn.net/u011109881/article/details/80379505>数据结构 Hash表（哈希表）
4. <https://zhuanlan.zhihu.com/p/144296454>图文并茂详解数据结构之哈希表
