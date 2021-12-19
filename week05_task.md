# 五、搜索,二叉堆

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----


@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 DFS相关题目

> **状态**，就是程序维护的所有动态数据构成的集合
> >所有**可能状态构成的集合**就是一个问题的**状态空间**



* 搜索
> 搜索就是采用直接遍历整个状态空间的方式寻找答案的一类算法 

根据遍历状态空间(图)方式的不同，可分为 
•深度优先搜索(DFS, depth first search) 
•广度优先搜索(BFS, breadth first search) 


一般来说，每个状态只遍历一次 
所以当状态空间是"图"而不是"树"时，要**判重(记忆化)**




### 1.1 [17 . 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)


* ==思路==:  建立数字到字母的映射,  用dfs搜索考虑每一种状态  >>> 注意共享变量还原

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic, ans = {}, []
        dic[2] = "abc"
        dic[3] = "def"
        dic[4] = "ghi"
        dic[5] = "jkl"
        dic[6] = "mno"
        dic[7] = "pqrs"
        dic[8] = "tuv"
        dic[9] = "wxyz"

        def dfs(index, strs):
            if index == len(digits):
                ans.append(strs)
                return 
            for i in dic[int(digits[index])]:
                dfs(index + 1, strs + i)
        if digits == '': return []
        dfs(0, "")
        return ans
```


### 1.2 [51 . N 皇后](https://leetcode-cn.com/problems/n-queens/)

* ==思路==: 选皇后位置>>> 即为全排列子集>>> 满足 `i - j`, `i + j`, `i`, `j`只出现一次


```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans, p, result = [], [], []
        used, usedPlus, usedminus = {}, {}, {}
        for i in range(n):
            used[i] = False

        for i in range(n):
            for j in range(n):
                usedPlus[i + j] = False
        
        for i in range(n):
            for j in range(n):
                usedminus[i - j] = False

        def dfs(row):
            if row == n:
                ans.append(p[:])
                print(ans)
                return
              
            for col in range(n):
                if not used[col] and not usedPlus[col + row] and not usedminus[col - row]: 
                    p.append(col)
                    used[col] = True
                    usedPlus[col + row] = True
                    usedminus[col - row] = True
                    dfs(row + 1)
                    usedminus[col - row] = False
                    usedPlus[col + row] = False
                    used[col] = False
                    p.pop()
        dfs(0)
        for p in ans:
            pattern = ["."*n] * n
            for row in range(n):
                pattern[row] = pattern[row][:p[row]] + "Q" + pattern[row][p[row] + 1 :]
            result.append(pattern)
        return result
```




## 2 BFS相关题目

### 2.1 [200 . 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

* ==思路==:  划分连通块 >>>BFS



```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n, ans = len(grid),len(grid[0]), 0
        visited = [[False] * n for _ in range(m)] 

        def bfs(sx, sy):
            q = []
            dx = [-1, 0 , 0, 1]  # 行的变化: 上  右  左 下
            dy = [0, 1, -1, 0]   # 列的变化: 上  右  左 下
            q.append([sx, sy])
            visited[sx][sy] = True
            while q:
                # 第一步取对头
                x = q[0][0]
                y = q[0][1]
                q.pop(0)
                # 扩展队头
                for i in range(4): # x y四个方向尝试走一走
                    nx = x + dx[i]
                    ny = y + dy[i]
                    if nx < 0 or nx >= m or ny >= n or ny < 0: continue #越界
                    if grid[nx][ny] != "1":continue                   # 不为1
                    if visited[nx][ny]: continue         #走过
                    q.append([nx, ny])
                    visited[nx][ny] = True

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1" and not visited[i][j]:
                    ans += 1
                    bfs(i, j)
        return ans
```



### 2.2 [433 . 最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/)

* ==思路==:  求转换步数 >>>> 图里求层数

```python
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        dic = {}
        dic[start] = 0
        if end not in bank:return -1
        else:
            q = []
            gene = ["A", "C", "G", "T"]
            q.append(start)
            while q:
                s = q[0]
                q.pop(0)
                for i in range(8):
                    for j in range(4):
                        if s[i] != gene[j]:
                            ns = s
                            ns = ns[:i] + gene[j] + ns[i + 1:]
                            if ns not in bank:continue          #改变的值不在基因库
                            if ns in dic:continue               #改变的值已经有过
                            dic[ns] = dic[s] + 1 
                            q.append(ns) 
                            if ns == end:
                                return dic[ns]
        return -1
```


### 2.3 [329 . 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

* ==法一 BFS==  递增 >>> 有向无环图 >>> 拓扑排序 >>> 可把一个点之前所有的路径计算并求其最大值:
    1. 先建图 (m*n)>>>  二维行列号变为一维:  `i * n + j `
    2. 开方向数组, 对合法两点间建边
    3. 从零入点出发, 拓扑排序 >>> 遍历出边 , 减度, 计算距离 >> 对入边为0点继续拓展
    4. 求最大值

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n, ans = len(matrix), len(matrix[0]), 0
        dx = [-1,0,0,1]
        dy = [0,1,-1,0]
        to, deg, q,dist = [[] for _ in range(m * n)], [0 for i in range(m * n)], [], [0 for i in range(m * n)]
        def num(i,j):
            return i * n + j

        def valid(x,y):
            return x >= 0 and x < m and y >= 0 and y < n

        def addEdge(u, v):       #拓扑排序 
            deg[v] += 1
            to[u].append(v)

        for i in range(m):
            for j in range(n):
                for k in range(4):
                    ni = i + dx[k]
                    nj = j + dy[k]
                    if valid(ni, nj) and matrix[ni][nj] > matrix[i][j] :
                        addEdge(num(i,j), num(ni, nj))
        for i in range(m * n):     #从零入度点出发
            if deg[i] == 0:
                q.append(i)
                dist[i] = 1
        while q:
            x = q[0]
            q.pop(0)
            for y in to[x]:   #遍历出边 减度
                deg[y] -= 1
                dist[y] = max(dist[y], dist[x] + 1)  #取x到y 加一步 与 dist[y]间的最大值
                if deg[y] == 0:  #当入边为0 继续扩展
                    q.append(y)

        for i in range(m * n):
            ans = max(ans, dist[i])
        return ans
```




* 法二 : DFS(记忆化搜索)


```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n, ans = len(matrix), len(matrix[0]), 0
        dx = [-1,0,0,1]
        dy = [0,1,-1,0]
        dist = [[0]*n for i in range(m)]
        def valid(x,y):
            return x >= 0 and x < m and y >= 0 and y < n

        def dfs(x, y):
            if dist[x][y] != 0:return dist[x][y]
            dist[x][y] = 1
            for k in range(4):
                nx = x + dx[k]
                ny = y + dy[k]
                if valid(nx, ny) and matrix[nx][ny] > matrix[x][y]:
                    dist[x][y] = max(dist[x][y], dfs(nx, ny) + 1)
            return dist[x][y]
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i , j))
        return ans
```


* 对比:
  * DFS更适合搜索树形状态空间 
•递归本身就会产生树的结构 
•可以用一个全局变量维护状态中较为复杂的信息（例：子集方案、排列方案） 
•不需要队列，节省空间
  * 求"最小代价"、"最少步数"的题目，用BFS 
•  BFS是按层次序搜索，第k步搜完才会搜k+1步，在任意时刻队列中至多只有两层
  * 状态空间为有向无环图 
•  BFS拓扑排序/ DFS记忆化搜索均可




## 3 二叉堆

>堆排序（Heap sort）基本思想：
>>借用「堆结构」所设计的排序算法。将数组转化为大顶堆，重复从大顶堆中取出数值最大的节点，并让剩余的堆维持大顶堆性质。

* 堆：符合以下两个条件之一的完全二叉树：

  * 大顶堆：根节点值 ≥ 子节点值。
  * 小顶堆：根节点值 ≤ 子节点值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f25869aa4d514f2c864ba302d00ae2e7.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
•本质上是一棵满足堆性质的完全二叉树 


* 常见操作 
  * 建堆(build) : O(N) 
  * 查询最值(get max/min) : O(1) 
  * 插入(insert) : O(log N) : 新元素一律插入到数组heap的尾部
  * 取出最值(delete max/min) : O(logN) :   把堆顶(heap[1])与堆尾(heap[n])交换，删除堆尾(数组最后一个元素) 


### 3.1 相关题目

1. [23 . 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

* ==思路==:  利用小顶堆,  依次弹出堆顶构建合法的链表

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq
        MinHeap = []
        for list_1 in lists:
            while list_1:
                heapq.heappush(MinHeap, list_1.val) #把listi中的数据逐个加到堆中
                list_1 = list_1.next
        dummy = ListNode(0) #构造虚节点
        p = dummy
        while MinHeap:
            p.next = ListNode(heapq.heappop(MinHeap)) #依次弹出最小堆的数据
            p = p.next
        return dummy.next 
```



2. [239 . 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)


* ==思路==: 懒惰删除法:  将删除操作待到影响窗口取值范围时, 再检查其是否过界删除


```python
class BinaryHeap:
    def __init__(self):
        self.heap = []
        
    def push(self, node):
        self.heap.append(node)
        for i in range((len(self.heap) - 2)//2, -1, -1): #从最后一个非叶子节点向上调整
            self.heapify(self.heap, i, len(self.heap) - 1)
    
    def pop(self):
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        for i in range((len(self.heap) - 2)//2 + 1):   #从0开始向下建立大顶堆
            self.heapify(self.heap, i, len(self.heap) - 1)


    # 调整完全二叉树>>>>大顶堆   
    def heapify(self, arr: [int], index: int, end: int ):
        left = index * 2 + 1                        #index的左右子节点
        right = left + 1
        while left <= end:                          #当index为非子节点时
            max_index = index
            if arr[left][0] > arr[max_index][0]:
                max_index = left
            if right <= end and arr[right][0] > arr[max_index][0]:
                max_index = right
            if index == max_index:                  #若不用交换，则说明已经交换结束
                break
            arr[index],arr[max_index] = arr[max_index],arr[index]
            index = max_index                      #继续调整index下一节点
            left = index * 2 + 1
            right = left + 1

            
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if len(nums) <= 1: return nums
        ans, q = [],BinaryHeap()
        for i in range(len(nums)):
            q.push([nums[i], i])
            if i >= k - 1:
                while q.heap[0][1] <= i - k:
                    q.pop()
                ans.append(q.heap[0][0])
        return ans
```




## 4 二叉搜索树

>二叉搜索树是一种节点值之间具有一定数量级次序的二叉树，对于树中每个节点：
>>若其左子树存在，则其**左子树**中每个节点的值都**不大于该节点值**；
>>若其右子树存在，则其**右子树**中每个节点的值都**不小于该节点值**。

![请添加图片描述](https://img-blog.csdnimg.cn/d1ffbad2827f4066b6028a75b85e56ef.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_10,color_FFFFFF,t_70,g_se,x_16)

**二叉搜索树的中序遍历必然为一个有序序列**



* **BST -建立** 
为了避免越界，减少边界情况的特殊判断，一般在BST中额外插入两个保护结点
 • 一个关键码为正无穷（一个很大的正整数） 
• 一个关键码为负无穷 
仅由这两个结点构成的BST就是一棵初始的空BST



![在这里插入图片描述](https://img-blog.csdnimg.cn/3118f398939c4b9d96b3abfca0368fd4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


* **BST -检索** 
检索关键码val是否存在 
从根开始递归查找 
•若当前结点的关键码等于vaL则已经找到 
•若关键码大于val,递归检索左子树（或不存在） 
•若关键码小于val,递归检索右子树（或不存在


* **BST -插入** 
插入val与检索val的过程类似 
•若检索发现存在，则放弃插入（或者把val对应结点的计数+1,视要求而定） 
•若检索发现不存在（子树为空）、直接在对应位置新建关键码为val的结点



* **BST -求前驱/后继** 
前驱：BST中小于val的最大结点 
后继：BST中大于val的最小结点 
求前驱和后继也是基于检索的，先检索val 
以后继为例： 
•如果检索到了 val,并且val存在右子树，则在**右子树中一直往左走到底** 
•否则说明**没找到val或者val没有右子树**，此时**后继就在检索过程经过的结点中** （即当前结点的所有祖先节点，可以拿一个变量顺便求一下）


* **BST-删除** 
从BST中删除关键码为val的结点，可以基于检索+求后继实现 
首先检索val 
如果val只有一棵子树，直接删除val,把子树和父结点相连就行了 
如果有两棵子树，需要找到后继，**先删除后继，再用后继结点代替val的位置** （因为后继是右子树一直往左走到底，所以它最多只会有一棵子树）


### 4.1 相关题目
1.[701 . 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)


* ==思路==: 插入基于检索 >>> 小往左, 大往右


```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:return TreeNode(val)
        if val < root.val:
            root.left = self.insertIntoBST(root.left, val)
        if val > root.val:
            root.right = self.insertIntoBST(root.right, val)
        return root
```

2.[面试题 04.06 . 后继者](https://leetcode-cn.com/problems/successor-lcci/)


* 思路:  基于检索:  找到val>> 右子树最左;   没找到 >>> 检索过程中


```python
class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
        return self.getnext(root, p.val)
    def getnext(self, root, val):
        ans = None
        cur = root
        while cur:
            if cur.val == val:
                if cur.right:   #存在val且右子树非空
                    ans = cur.right
                    while ans.left:ans = ans.left
                break
            
            if val < cur.val:
                if not ans or ans.val > cur.val:  #检索过程中在比val大的搜索过程, 更新min
                    ans = cur
                cur = cur.left
            else:
                cur = cur.right
        return ans
```


3.[450 . 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)


* ==思路==: 先检索:	找到val :无左返回右, 无右返回左, 都有用后继替换


```python
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if root.val == key:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            Successor = root.right  #有两个子节点 : 寻找后继
            while Successor.left:
                Successor = Successor.left
            root.right = self.deleteNode(root.right, Successor.val)
            root.val = Successor.val
            return root
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            root.left = self.deleteNode(root.left, key)
        return root
```



## 习题


### 1  [130 . 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)


* ==思路一== dfs >>>  搜索出四周边的"O"及相连"O" >>> 即为未被包围的标记为"P" >>> 搜索其他"O"修改为"x"


```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m, n = len(board), len(board[0])

        def dfs(x, y):
            if board[x][y] != "O":return
            else:
                board[x][y] = "Q"
            dx = [-1, 0, 1, 0]
            dy = [0, 1, 0, -1]
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if 0 <= nx < m and 0 <= ny < n:
                    dfs(nx, ny)

        for row in range(m):  #第一列 及最后一列搜索"O"及相连"O"
            dfs(row, 0)
            dfs(row, n - 1)
        for col in range(1, n - 1): # 第一行及最后一行
            dfs(0, col)
            dfs(m - 1, col)
        for i in range(m):
            for j in range(n):
                board[i][j] = 'O' if board[i][j] == 'Q' else 'X'
```


* ==思路二==: 类似的利用BFS


```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m, n = len(board), len(board[0])

        def bfs(sx, sy):
            from collections import deque
            q = deque()
            board[sx][sy] = "Q"
            q.append((sx, sy))
            while q:
                x, y = q.popleft()
                dx = [-1, 0, 1, 0]
                dy = [0, 1, 0, -1]
                for i in range(4):
                    nx = x + dx[i]
                    ny = y + dy[i]
                    if 0 <= nx < m and 0 <= ny < n and board[nx][ny] == "O":
                        board[nx][ny] = "Q"
                        q.append((nx, ny))

        for row in range(m):  #第一列 及最后一列搜索"O"及相连"O"
            if board[row][0] == "O":
                bfs(row, 0)
            if board[row][n - 1] == "O":
                bfs(row, n - 1)
        for col in range(1, n - 1): # 第一行及最后一行
            if board[m - 1][col] == "O":
                bfs(m - 1, col)
            if board[0][col] == "O":
                bfs(0, col)

        for i in range(m):
            for j in range(n):
                board[i][j] = 'O' if board[i][j] == 'Q' else 'X'
```







2. [538 . 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)


* ==思路==: dfs

```python
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root, sumBST):
            if not root:
                return 0
            if root.right:
                sumBST = dfs(root.right, sumBST)
            root.val += sumBST
            if root.left:
                sumBST = dfs(root.left, root.val)
                return sumBST
            else:
                return root.val
        dfs(root, 0)
        return root
```

## 参考资料
1.<https://www.jianshu.com/p/ff4b93b088eb>(数据结构（二）：二叉搜索树（Binary Search Tree）)
