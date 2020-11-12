const std = @import("std");
const math = std.math;
const stdout = std.io.getStdOut().outStream();
const testing = std.testing;

pub fn Graph(comptime NodeValueT: type, comptime EdgeValueT: type) type {
    return struct {
        const Self = @This();
        nodes: std.ArrayList(*Node),
        allocator: *std.mem.Allocator,

        // We can probably pull this out of graph and just have it as a static type
        const Edge = struct {
            in: usize,
            out: usize,
            weight: EdgeValueT,

            pub fn init(in: usize, out: usize, weight: EdgeValueT) Edge {
                return Edge{
                    .in = in,
                    .out = out,
                    .weight = weight,
                };
            }
        };

        const Node = struct {
            value: NodeValueT,
            edges: std.ArrayList(*Edge),
            // allocator: *std.mem.Allocator,

            pub fn init(value: NodeValueT, alloc: *std.mem.Allocator) Node {
                return Node{
                    .value = value,
                    .edges = std.ArrayList(*Edge).init(alloc),
                    // .allocator = alloc
                };
            }

            pub fn addEdge(self: *Node, edge: *Edge) !void {
                try self.edges.append(edge);
            }
        };

        pub fn init(alloc: *std.mem.Allocator) Self {
            return Self{
                .nodes = std.ArrayList(*Node).init(alloc),
                .allocator = alloc,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.nodes.items) |node| {
                for (node.edges.items) |edge| {
                    self.allocator.destroy(edge);
                }
                node.edges.deinit();
                self.allocator.destroy(node);
            }
            self.nodes.deinit();
        }

        pub fn addNode(self: *Self, n: NodeValueT) !usize {
            var node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(node);
            node.* = Node.init(n, self.allocator);
            try self.nodes.append(node);
            return self.nodes.items.len - 1; // The index of the added node
        }

        pub fn addEdge(self: *Self, from_idx: usize, to_idx: usize, weight: EdgeValueT) !void {
            var from = self.nodes.items[from_idx];
            var to = self.nodes.items[to_idx];
            // TODO throw error if from or to is out of range

            var edge = try self.allocator.create(Edge);
            errdefer self.allocator.destroy(edge);
            edge.* = Edge.init(from_idx, to_idx, weight);
            try from.addEdge(edge);
        }

        pub fn addEdgeBidirectional(self: *Self, from_idx: usize, to_idx: usize, weight: EdgeValueT) !void {
            try self.addEdge(from_idx, to_idx, weight);
            try self.addEdge(to_idx, from_idx, weight);
        }

        pub fn print(self: *Self) !void {
            try stdout.print("# Nodes: {}\n", .{self.nodes.items.len});
            for (self.nodes.items) |node, idx| {
                try stdout.print("{}: {}\n", .{ idx, node.value });
                for (node.edges.items) |edge| {
                    try stdout.print("\t{} -> {} ({})\n", .{ edge.in, edge.out, edge.weight });
                }
            }
        }

        const AStarPQueueElement = struct {
            curr_index: usize,
            prev_index: usize, //TODO should make this optional. currently overloading node == prev
            cost_to_come: EdgeValueT, //g
            cost_to_go: EdgeValueT, //h

            pub fn init(curr_index: usize, prev_index: usize, cost_to_come: EdgeValueT, cost_to_go: EdgeValueT) AStarPQueueElement {
                return AStarPQueueElement{
                    .curr_index = curr_index,
                    .prev_index = prev_index,
                    .cost_to_come = cost_to_come,
                    .cost_to_go = cost_to_go,
                };
            }
        };

        fn AstarPQueueElementComparator(a: AStarPQueueElement, b: AStarPQueueElement) bool {
            return a.cost_to_come + a.cost_to_go < b.cost_to_come + b.cost_to_go;
        }

        const AStarVistiedElement = struct {
            prev_index: usize,
            cost_to_come: EdgeValueT,

            pub fn init(prev_index: usize, cost_to_come: EdgeValueT) AStarVistiedElement {
                return AStarVistiedElement{
                    .prev_index = prev_index,
                    .cost_to_come = cost_to_come,
                };
            }
        };

        const AStarResult = struct {
            path: std.ArrayList(usize),
            path_len: EdgeValueT,
            visited_count: usize,
            allocator: *std.mem.Allocator,

            pub fn init(alloc: *std.mem.Allocator, path: std.ArrayList(usize), path_len: EdgeValueT, visited_count: usize) AStarResult {
                return AStarResult{
                    .path = path,
                    .path_len = path_len,
                    .visited_count = visited_count,
                    .allocator = alloc,
                };
            }

            pub fn print(self: *AStarResult) !void {
                try stdout.print("Nodes Visited: {}\n", .{self.visited_count});
                try stdout.print("Path Length: {}\n", .{self.path_len});
                // try stdout.print("Path: {}\n", .{self.path.items});
                for (self.path.items) |path_idx, idx| {
                    try stdout.print("{}", .{path_idx});
                    if (idx < self.path.items.len - 1) {
                        try stdout.print("->", .{});
                    }
                }
                try stdout.print("\n", .{});
            }

            pub fn deinit(self: *AStarResult) void {
                self.path.deinit();
            }
        };

        fn extractAStarSolution(self: *Self, start_idx: usize, goal_idx: usize, visited: std.AutoHashMap(usize, AStarVistiedElement), num_visited: usize) !AStarResult {
            var path = std.ArrayList(usize).init(self.allocator);
            try path.append(goal_idx);

            var prev = visited.get(goal_idx).?.prev_index;
            // TODO check if solution found lol
            while (prev != start_idx) {
                const curr = prev;
                try path.append(curr);
                prev = visited.get(curr).?.prev_index;
            }
            try path.append(start_idx);
            std.mem.reverse(usize, path.items);

            // var it = visited.iterator();
            // while (it.next()) |kv| {
            //     std.debug.warn("{}: {}\n", .{ kv.key, kv.value });
            // }

            return AStarResult.init(self.allocator, path, visited.get(goal_idx).?.cost_to_come, num_visited);
        }

        pub fn AStarSearch(self: *Self, start_idx: usize, goal_idx: usize, h: fn (n1: NodeValueT, n2: NodeValueT) EdgeValueT) !AStarResult {
            const start = self.nodes.items[start_idx].value;
            const goal = self.nodes.items[goal_idx].value;

            // Create priority queue
            var pq = std.PriorityQueue(AStarPQueueElement).init(self.allocator, AstarPQueueElementComparator);
            defer pq.deinit();

            // Create visited map
            var visited = std.AutoHashMap(usize, AStarVistiedElement).init(self.allocator);
            defer visited.deinit();

            var num_visited: usize = 0;

            // Add start node
            try pq.add(AStarPQueueElement.init(start_idx, start_idx, 0.0, h(start, goal)));

            // Start search
            while (pq.count() != 0) {
                const best = pq.remove();
                num_visited += 1;
                const found = visited.contains(best.curr_index);
                const better = blk: {
                    if (found and best.cost_to_come < visited.get(best.curr_index).?.cost_to_come) {
                        break :blk true;
                    }
                    break :blk false;
                };
                // If we haven't seen this node before, or we get here cheaper
                if (!found or better) {
                    // add to visited
                    try visited.put(best.curr_index, AStarVistiedElement.init(best.prev_index, best.cost_to_come));

                    if (best.curr_index == goal_idx) {
                        // std.debug.warn("Found that shit\n", .{});
                        break;
                    }

                    // loop over children
                    const parent = self.nodes.items[best.curr_index];
                    for (parent.edges.items) |edge| {
                        const child_cost_to_come = best.cost_to_come + edge.weight;
                        // check if the child has been explored
                        const child_found = visited.contains(edge.out);
                        const child_better = blk: {
                            if (child_found and child_cost_to_come < visited.get(edge.out).?.cost_to_come) {
                                break :blk true;
                            }
                            break :blk false;
                        };

                        if (!child_found or child_better) {
                            const child = self.nodes.items[edge.out].value;
                            try pq.add(AStarPQueueElement.init(edge.out, best.curr_index, child_cost_to_come, h(child, goal)));
                        }
                    }
                }
            }
            return try self.extractAStarSolution(start_idx, goal_idx, visited, num_visited);
        }
    };
}

// :( can't figure out how to use the built in vector type
pub const Vec2 = struct {
    x: f32,
    y: f32,
};

pub fn euclidean(v1: Vec2, v2: Vec2) f32 {
    return math.sqrt(math.pow(f32, (v1.x - v2.x), 2) + math.pow(f32, (v1.y - v2.y), 2));
}

test "Simple AStar Problem" {

    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // _ = gpa.deinit();
    // var allocator = &gpa.allocator;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = &arena.allocator;

    var graph = Graph(Vec2, f32).init(allocator);
    defer graph.deinit();

    // 2 | 5 | 8
    // --+---+--
    // 1 | 4 | 7
    // --+---+--
    // 0 | 3 | 6
    var x: f32 = -1;
    var y: f32 = -1;
    while (x <= 1) {
        while (y <= 1) : (y += 1) {
            _ = try graph.addNode(Vec2{ .x = x, .y = y });
        }
        y = -1;
        x += 1;
    }

    // TODO: Fix hardcoding of distance
    // Can create slice of nodes first in a block and then add to graph

    // Add 4 connected edges
    // Y direction edges
    try graph.addEdgeBidirectional(0, 1, 1.0);
    try graph.addEdgeBidirectional(1, 2, 1.0);
    try graph.addEdgeBidirectional(3, 4, 1.0);
    try graph.addEdgeBidirectional(4, 5, 1.0);
    try graph.addEdgeBidirectional(6, 7, 1.0);
    try graph.addEdgeBidirectional(7, 8, 1.0);

    // // X-direction edges
    try graph.addEdgeBidirectional(0, 3, 1.0);
    try graph.addEdgeBidirectional(3, 6, 1.0);
    try graph.addEdgeBidirectional(1, 4, 1.0);
    try graph.addEdgeBidirectional(4, 7, 1.0);
    try graph.addEdgeBidirectional(2, 5, 1.0);
    try graph.addEdgeBidirectional(5, 8, 1.0);

    // Add Diagonal edges
    try graph.addEdgeBidirectional(0, 4, std.math.sqrt(2.0));
    try graph.addEdgeBidirectional(1, 5, std.math.sqrt(2.0));
    try graph.addEdgeBidirectional(3, 7, std.math.sqrt(2.0));
    try graph.addEdgeBidirectional(4, 8, std.math.sqrt(2.0));

    try graph.addEdgeBidirectional(1, 3, std.math.sqrt(2.0));
    try graph.addEdgeBidirectional(2, 4, std.math.sqrt(2.0));
    try graph.addEdgeBidirectional(4, 6, std.math.sqrt(2.0));
    try graph.addEdgeBidirectional(5, 7, std.math.sqrt(2.0));

    var result = try graph.AStarSearch(0, 8, euclidean);
    defer result.deinit();

    testing.expect(result.visited_count == 3);
    testing.expect(result.path_len == 2.0 * std.math.sqrt(2.0));
    // TODO not quite sure how to check that the path is identical
    // testing.expect(std.mem.eql([3]usize,result.path.toOwnedSlice(),[3]usize{0,4,8}));
}
